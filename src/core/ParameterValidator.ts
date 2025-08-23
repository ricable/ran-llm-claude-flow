import { Parameter, ValidationRule, ParameterType } from '../types';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

/**
 * Parameter validation engine with extensible rules
 * Supports range, enum, pattern, and custom validation
 */
export class ParameterValidator {
  private logger: Logger;
  private customValidators: Map<string, (value: any, rule: any) => boolean> = new Map();
  private parameterRules: Map<string, ValidationRule[]> = new Map();

  constructor() {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      defaultMeta: { service: 'ParameterValidator' },
      transports: [
        new transports.Console({
          format: format.combine(
            format.colorize(),
            format.simple()
          )
        })
      ]
    });

    this.initializeDefaultRules();
  }

  /**
   * Validate a parameter against its rules
   */
  public async validateParameter(parameter: Parameter): Promise<void> {
    this.logger.debug('Validating parameter', { 
      name: parameter.name, 
      type: parameter.type, 
      value: parameter.value 
    });

    // Get validation rules
    const rules = parameter.validationRules || this.parameterRules.get(parameter.name) || [];
    
    // Type-specific validation
    await this.validateByType(parameter);

    // Apply all validation rules
    for (const rule of rules) {
      const isValid = await this.applyRule(parameter.value, rule);
      if (!isValid) {
        const error = `Parameter ${parameter.name} validation failed: ${rule.message}`;
        this.logger.error('Validation failed', { parameter: parameter.name, rule: rule.type, error });
        throw new Error(error);
      }
    }

    this.logger.debug('Parameter validation passed', { name: parameter.name });
  }

  /**
   * Validate multiple parameters
   */
  public async validateParameters(parameters: Parameter[]): Promise<void> {
    const errors: string[] = [];

    for (const parameter of parameters) {
      try {
        await this.validateParameter(parameter);
      } catch (error) {
        errors.push(`${parameter.name}: ${error}`);
      }
    }

    if (errors.length > 0) {
      const errorMessage = `Parameter validation failed:\n${errors.join('\n')}`;
      this.logger.error('Multiple parameter validation failed', { errors });
      throw new Error(errorMessage);
    }
  }

  /**
   * Register a custom validation rule for a parameter
   */
  public registerParameterRule(parameterName: string, rule: ValidationRule): void {
    if (!this.parameterRules.has(parameterName)) {
      this.parameterRules.set(parameterName, []);
    }
    this.parameterRules.get(parameterName)!.push(rule);
    this.logger.info('Parameter rule registered', { parameterName, ruleType: rule.type });
  }

  /**
   * Register a custom validator function
   */
  public registerCustomValidator(
    name: string,
    validator: (value: any, rule: any) => boolean
  ): void {
    this.customValidators.set(name, validator);
    this.logger.info('Custom validator registered', { name });
  }

  /**
   * Get validation rules for a parameter
   */
  public getParameterRules(parameterName: string): ValidationRule[] {
    return this.parameterRules.get(parameterName) || [];
  }

  /**
   * Type-specific validation
   */
  private async validateByType(parameter: Parameter): Promise<void> {
    switch (parameter.type) {
      case ParameterType.RF:
        await this.validateRFParameter(parameter);
        break;
      case ParameterType.MOBILITY:
        await this.validateMobilityParameter(parameter);
        break;
      case ParameterType.HANDOVER:
        await this.validateHandoverParameter(parameter);
        break;
      case ParameterType.POWER:
        await this.validatePowerParameter(parameter);
        break;
      case ParameterType.ANTENNA:
        await this.validateAntennaParameter(parameter);
        break;
      case ParameterType.CARRIER:
        await this.validateCarrierParameter(parameter);
        break;
      default:
        // Generic validation
        break;
    }
  }

  /**
   * Apply a validation rule
   */
  private async applyRule(value: any, rule: ValidationRule): Promise<boolean> {
    switch (rule.type) {
      case 'range':
        return this.validateRange(value, rule.rule);
      case 'enum':
        return this.validateEnum(value, rule.rule);
      case 'pattern':
        return this.validatePattern(value, rule.rule);
      case 'custom':
        return this.validateCustom(value, rule.rule);
      default:
        this.logger.warn('Unknown validation rule type', { type: rule.type });
        return true;
    }
  }

  /**
   * Range validation
   */
  private validateRange(value: any, range: { min: number; max: number }): boolean {
    const numValue = Number(value);
    if (isNaN(numValue)) {
      return false;
    }
    return numValue >= range.min && numValue <= range.max;
  }

  /**
   * Enum validation
   */
  private validateEnum(value: any, enumValues: any[]): boolean {
    return enumValues.includes(value);
  }

  /**
   * Pattern validation
   */
  private validatePattern(value: any, pattern: string | RegExp): boolean {
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    return regex.test(String(value));
  }

  /**
   * Custom validation
   */
  private validateCustom(value: any, rule: any): boolean {
    if (typeof rule === 'string' && this.customValidators.has(rule)) {
      return this.customValidators.get(rule)!(value, rule);
    }
    if (typeof rule === 'function') {
      return rule(value);
    }
    this.logger.warn('Invalid custom validation rule', { rule });
    return true;
  }

  /**
   * RF parameter validation
   */
  private async validateRFParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    // Common RF parameter validations
    if (name.toLowerCase().includes('frequency')) {
      if (typeof value !== 'number' || value <= 0) {
        throw new Error(`${name} must be a positive number`);
      }
    }
    
    if (name.toLowerCase().includes('power')) {
      if (typeof value !== 'number') {
        throw new Error(`${name} must be a number`);
      }
    }
  }

  /**
   * Mobility parameter validation
   */
  private async validateMobilityParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    if (name.toLowerCase().includes('threshold')) {
      if (typeof value !== 'number') {
        throw new Error(`${name} must be a number`);
      }
    }
    
    if (name.toLowerCase().includes('hysteresis')) {
      if (typeof value !== 'number' || value < 0) {
        throw new Error(`${name} must be a non-negative number`);
      }
    }
  }

  /**
   * Handover parameter validation
   */
  private async validateHandoverParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    if (name.toLowerCase().includes('timer')) {
      if (typeof value !== 'number' || value <= 0) {
        throw new Error(`${name} must be a positive number`);
      }
    }
  }

  /**
   * Power parameter validation
   */
  private async validatePowerParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    if (typeof value !== 'number') {
      throw new Error(`${name} must be a number`);
    }
    
    // Common power ranges (dBm)
    if (value < -50 || value > 50) {
      this.logger.warn('Power value outside typical range', { 
        name, 
        value, 
        expectedRange: '[-50, 50] dBm' 
      });
    }
  }

  /**
   * Antenna parameter validation
   */
  private async validateAntennaParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    if (name.toLowerCase().includes('tilt')) {
      if (typeof value !== 'number') {
        throw new Error(`${name} must be a number`);
      }
      if (Math.abs(value) > 90) {
        throw new Error(`${name} tilt must be between -90 and 90 degrees`);
      }
    }
    
    if (name.toLowerCase().includes('azimuth')) {
      if (typeof value !== 'number') {
        throw new Error(`${name} must be a number`);
      }
      if (value < 0 || value >= 360) {
        throw new Error(`${name} must be between 0 and 359 degrees`);
      }
    }
  }

  /**
   * Carrier parameter validation
   */
  private async validateCarrierParameter(parameter: Parameter): Promise<void> {
    const { name, value } = parameter;
    
    if (name.toLowerCase().includes('bandwidth')) {
      if (typeof value !== 'number' || value <= 0) {
        throw new Error(`${name} must be a positive number`);
      }
    }
  }

  /**
   * Initialize default validation rules
   */
  private initializeDefaultRules(): void {
    // Common LTE parameters
    this.registerParameterRule('cellBarred', {
      type: 'enum',
      rule: ['barred', 'notBarred'],
      message: 'cellBarred must be either "barred" or "notBarred"'
    });

    this.registerParameterRule('tac', {
      type: 'range',
      rule: { min: 1, max: 65535 },
      message: 'TAC must be between 1 and 65535'
    });

    this.registerParameterRule('pci', {
      type: 'range',
      rule: { min: 0, max: 503 },
      message: 'PCI must be between 0 and 503'
    });

    // 5G NR parameters
    this.registerParameterRule('nci', {
      type: 'range',
      rule: { min: 0, max: 68719476735 }, // 36 bits
      message: 'NCI must be a valid 36-bit value'
    });

    this.registerParameterRule('ssbSubCarrierSpacing', {
      type: 'enum',
      rule: [15, 30, 120, 240],
      message: 'SSB subcarrier spacing must be 15, 30, 120, or 240 kHz'
    });

    this.logger.info('Default validation rules initialized');
  }
}