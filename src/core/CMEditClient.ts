import { CMEditCommand, CMEditResponse } from '../types';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { v4 as uuidv4 } from 'uuid';

/**
 * CMEDIT client for executing commands on Ericsson network elements
 * Provides REST API interface with connection pooling and retry logic
 */
export class CMEditClient {
  private logger: Logger;
  private httpClient: AxiosInstance;
  private connected: boolean = false;
  private baseUrl: string;
  private credentials?: {
    username: string;
    password: string;
  };
  private sessionToken?: string;
  private retryConfig = {
    maxRetries: 3,
    retryDelay: 1000, // ms
    backoffMultiplier: 2
  };

  constructor(
    baseUrl: string = 'https://ericsson-oss.local',
    credentials?: { username: string; password: string }
  ) {
    this.baseUrl = baseUrl;
    this.credentials = credentials;

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'CMEditClient' },
      transports: [
        new transports.Console({
          format: format.combine(
            format.colorize(),
            format.simple()
          )
        })
      ]
    });

    this.httpClient = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });

    this.setupInterceptors();
  }

  /**
   * Connect to the CMEDIT service
   */
  public async connect(): Promise<void> {
    if (this.connected) {
      this.logger.info('Already connected to CMEDIT service');
      return;
    }

    try {
      this.logger.info('Connecting to CMEDIT service', { baseUrl: this.baseUrl });

      if (this.credentials) {
        await this.authenticate();
      }

      // Test connection
      await this.testConnection();
      
      this.connected = true;
      this.logger.info('Successfully connected to CMEDIT service');

    } catch (error) {
      this.logger.error('Failed to connect to CMEDIT service', { error });
      throw new Error(`CMEDIT connection failed: ${error}`);
    }
  }

  /**
   * Disconnect from the CMEDIT service
   */
  public async disconnect(): Promise<void> {
    if (!this.connected) {
      return;
    }

    try {
      this.logger.info('Disconnecting from CMEDIT service');
      
      if (this.sessionToken) {
        await this.logout();
      }
      
      this.connected = false;
      this.sessionToken = undefined;
      
      this.logger.info('Disconnected from CMEDIT service');
    } catch (error) {
      this.logger.error('Error during disconnect', { error });
    }
  }

  /**
   * Execute a CMEDIT command
   */
  public async execute(command: CMEditCommand): Promise<CMEditResponse> {
    if (!this.connected) {
      await this.connect();
    }

    const commandId = uuidv4();
    const timestamp = new Date();

    this.logger.info('Executing CMEDIT command', { 
      commandId, 
      operation: command.operation, 
      mo: command.mo 
    });

    try {
      const response = await this.executeWithRetry(command, commandId);
      
      const cmResponse: CMEditResponse = {
        success: true,
        data: response.data,
        commandId,
        timestamp
      };

      this.logger.info('CMEDIT command executed successfully', { commandId });
      return cmResponse;

    } catch (error) {
      const cmResponse: CMEditResponse = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        commandId,
        timestamp
      };

      this.logger.error('CMEDIT command failed', { commandId, error });
      return cmResponse;
    }
  }

  /**
   * Execute multiple commands in batch
   */
  public async executeBatch(commands: CMEditCommand[]): Promise<CMEditResponse[]> {
    this.logger.info('Executing batch CMEDIT commands', { count: commands.length });

    const results: CMEditResponse[] = [];
    
    // Execute commands sequentially to maintain order
    for (const command of commands) {
      const result = await this.execute(command);
      results.push(result);
      
      // Stop on first failure if required
      if (!result.success && this.shouldStopOnFailure(command)) {
        this.logger.warn('Batch execution stopped due to failure', { 
          commandId: result.commandId 
        });
        break;
      }
    }

    this.logger.info('Batch execution completed', { 
      total: commands.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length
    });

    return results;
  }

  /**
   * Get managed object information
   */
  public async getManagedObject(mo: string, attributes?: string[]): Promise<any> {
    const command: CMEditCommand = {
      operation: 'get',
      mo,
      ...(attributes && { attributes: { select: attributes } })
    };

    const response = await this.execute(command);
    
    if (!response.success) {
      throw new Error(`Failed to get MO ${mo}: ${response.error}`);
    }

    return response.data;
  }

  /**
   * Create managed object
   */
  public async createManagedObject(
    mo: string, 
    attributes: Record<string, any>
  ): Promise<any> {
    const command: CMEditCommand = {
      operation: 'create',
      mo,
      attributes
    };

    const response = await this.execute(command);
    
    if (!response.success) {
      throw new Error(`Failed to create MO ${mo}: ${response.error}`);
    }

    return response.data;
  }

  /**
   * Update managed object
   */
  public async updateManagedObject(
    mo: string, 
    attributes: Record<string, any>
  ): Promise<any> {
    const command: CMEditCommand = {
      operation: 'update',
      mo,
      attributes
    };

    const response = await this.execute(command);
    
    if (!response.success) {
      throw new Error(`Failed to update MO ${mo}: ${response.error}`);
    }

    return response.data;
  }

  /**
   * Delete managed object
   */
  public async deleteManagedObject(mo: string): Promise<void> {
    const command: CMEditCommand = {
      operation: 'delete',
      mo
    };

    const response = await this.execute(command);
    
    if (!response.success) {
      throw new Error(`Failed to delete MO ${mo}: ${response.error}`);
    }
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.connected;
  }

  /**
   * Execute command with retry logic
   */
  private async executeWithRetry(
    command: CMEditCommand, 
    commandId: string,
    attempt: number = 1
  ): Promise<any> {
    try {
      const endpoint = this.getEndpointForOperation(command.operation);
      const requestData = this.buildRequestData(command);
      
      const config: AxiosRequestConfig = {
        headers: {
          'X-Command-ID': commandId,
          ...(this.sessionToken && { 'Authorization': `Bearer ${this.sessionToken}` })
        }
      };

      let response;
      switch (command.operation) {
        case 'get':
          response = await this.httpClient.get(endpoint, { ...config, params: requestData });
          break;
        case 'create':
        case 'update':
          response = await this.httpClient.post(endpoint, requestData, config);
          break;
        case 'delete':
          response = await this.httpClient.delete(endpoint, { ...config, data: requestData });
          break;
        default:
          throw new Error(`Unsupported operation: ${command.operation}`);
      }

      return response;

    } catch (error) {
      if (attempt < this.retryConfig.maxRetries && this.shouldRetry(error)) {
        const delay = this.retryConfig.retryDelay * Math.pow(this.retryConfig.backoffMultiplier, attempt - 1);
        
        this.logger.warn('Command failed, retrying', { 
          commandId, 
          attempt, 
          delay, 
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        
        await this.sleep(delay);
        return this.executeWithRetry(command, commandId, attempt + 1);
      }
      
      throw error;
    }
  }

  /**
   * Authenticate with the service
   */
  private async authenticate(): Promise<void> {
    if (!this.credentials) {
      throw new Error('No credentials provided for authentication');
    }

    try {
      const response = await this.httpClient.post('/auth/login', {
        username: this.credentials.username,
        password: this.credentials.password
      });

      this.sessionToken = response.data.token;
      this.logger.info('Authentication successful');

    } catch (error) {
      this.logger.error('Authentication failed', { error });
      throw new Error('CMEDIT authentication failed');
    }
  }

  /**
   * Logout and clear session
   */
  private async logout(): Promise<void> {
    try {
      await this.httpClient.post('/auth/logout', {}, {
        headers: { 'Authorization': `Bearer ${this.sessionToken}` }
      });
      this.logger.info('Logout successful');
    } catch (error) {
      this.logger.warn('Logout failed', { error });
    }
  }

  /**
   * Test connection to the service
   */
  private async testConnection(): Promise<void> {
    try {
      await this.httpClient.get('/health');
    } catch (error) {
      throw new Error('CMEDIT service health check failed');
    }
  }

  /**
   * Setup HTTP interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.httpClient.interceptors.request.use(
      (config) => {
        this.logger.debug('HTTP request', { 
          method: config.method?.toUpperCase(), 
          url: config.url 
        });
        return config;
      },
      (error) => {
        this.logger.error('HTTP request error', { error });
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.httpClient.interceptors.response.use(
      (response) => {
        this.logger.debug('HTTP response', { 
          status: response.status, 
          url: response.config.url 
        });
        return response;
      },
      (error) => {
        this.logger.error('HTTP response error', { 
          status: error.response?.status,
          url: error.config?.url,
          error: error.message
        });
        return Promise.reject(error);
      }
    );
  }

  /**
   * Get API endpoint for operation
   */
  private getEndpointForOperation(operation: string): string {
    const endpoints = {
      get: '/cmedit/objects',
      create: '/cmedit/objects',
      update: '/cmedit/objects',
      delete: '/cmedit/objects'
    };
    
    return endpoints[operation as keyof typeof endpoints] || '/cmedit/objects';
  }

  /**
   * Build request data for command
   */
  private buildRequestData(command: CMEditCommand): any {
    return {
      operation: command.operation,
      mo: command.mo,
      attributes: command.attributes,
      filter: command.filter
    };
  }

  /**
   * Check if error should trigger retry
   */
  private shouldRetry(error: any): boolean {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      // Retry on network errors, timeouts, and 5xx errors
      return !status || status >= 500 || status === 408;
    }
    return false;
  }

  /**
   * Check if batch should stop on command failure
   */
  private shouldStopOnFailure(command: CMEditCommand): boolean {
    // Stop on delete failures to avoid cascading issues
    return command.operation === 'delete';
  }

  /**
   * Sleep utility for retry delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}