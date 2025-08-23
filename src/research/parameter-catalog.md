# RAN Parameter Catalog Research Report

## Executive Summary

This document provides a comprehensive analysis of Ericsson RAN parameters identified from the codebase, categorizing them by function, technology, and usage patterns to inform SDK parameter management design.

## 1. Parameter Classification Framework

### 1.1 Hierarchical Parameter Structure
Based on the CSV processing analysis, parameters follow a consistent hierarchical naming convention:

```
[Model].[MOClass].[ParameterName]
Example: ManagedElement.EUtranCellFDD.qRxLevMin
```

### 1.2 Parameter Categories
From `csv/types.rs` analysis:

```rust
pub enum ParameterCategory {
    System,              // Core system parameters
    Network,             // Network configuration  
    Performance,         // Performance tuning
    Security,            // Security settings
    QualityOfService,    // QoS management
    CarrierAggregation,  // CA configuration
    Mobility,            // Handover and mobility
    AccessControl,       // Access management
    Measurement,         // Measurement configuration
    Other,               // Uncategorized
}
```

## 2. Core System Parameters

### 2.1 ManagedElement Parameters
**Identity and Basic Configuration:**

| Parameter | Type | Description | Usage |
|-----------|------|-------------|-------|
| `managedElementId` | string | MO identifier | Read-only system identifier |
| `siteLocation` | string | Geographic location | Optional descriptive field |
| `userLabel` | string | User description | Optional identification aid |
| `managedElementType` | string | Product type (RBS/CSCF) | System-generated |
| `release` | string | Software release | System information |
| `swVersion` | string | Software version | 3GPP alignment |

**Network Integration:**

| Parameter | Type | Description | Range/Values |
|-----------|------|-------------|--------------|
| `dnPrefix` | string | Distinguished Name prefix | Domain partitioning |
| `networkManagedElementId` | string | Network unique ID | 3GPP naming convention |
| `timeZone` | string | Time zone (deprecated) | Olson database format |
| `localDateTime` | string | Local date/time (deprecated) | YYYY-MM-DDThh:mm:ss |

### 2.2 SystemFunctions Parameters
| Parameter | Description |
|-----------|-------------|
| `systemFunctionsId` | System functions MO identifier |

### 2.3 Transport Parameters
| Parameter | Description |
|-----------|-------------|  
| `transportId` | Transport layer MO identifier |

## 3. Radio Access Network Parameters

### 3.1 4G/LTE Parameters (EUtranCell)

#### Cell Selection and Reselection
| Parameter | Type | Description | Trigger Formula |
|-----------|------|-------------|-----------------|
| `qRxLevMin` | int32 | Minimum RSRP level | `RSRP > EUtranCellFDD.qRxLevMin` |
| `qQualMin` | int32 | Minimum RSRQ level | `RSRQ > EUtranCellFDD.qQualMin` |
| `qRxLevMinOffset` | int32 | PLMN offset for qRxLevMin | Default: 1000 (not sent) |
| `qQualMinOffset` | int32 | PLMN offset for qQualMin | Default: 0 |

#### Mobility Management
| Parameter | Type | Description | Usage Pattern |
|-----------|------|-------------|---------------|
| `a1a2UlSearchThreshold` | int32 | UL measurement threshold | Mobility trigger calculation |
| `sIntraSearch` | int32 | Intra-frequency search | 1000 = not sent |
| `pMaxServingCell` | int32 | Max UE power in serving cell | Default: 1000 |

### 3.2 5G/NR Parameters

#### GNBCUCPFunction Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `nRTAC` | int32 | NR Tracking Area Code |
| `pLMNIdList` | struct | List of supported PLMNs (1-12) |

#### NRCellDU Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `pMax` | int32 | Maximum UE transmit power | 23 dBm |

#### AdmissionControl Parameters
| Parameter | Description |
|-----------|-------------|
| `admissionControlId` | Admission control MO identifier |

## 4. Feature Management Parameters

### 4.1 FeatureState Pattern
**Critical Parameter**: `FeatureState.featureState`
- **Type**: Enumerated/Boolean
- **Function**: Controls feature activation/deactivation  
- **Usage**: Central to all feature lifecycle management
- **Pattern**: Appears consistently across all feature configurations

```yaml
feature_activation_pattern:
  current_state: "DEACTIVATED"
  desired_state: "ACTIVATED"
  transition_validation: required
  dependencies_check: mandatory
  rollback_capability: essential
```

### 4.2 Feature Categories
Based on analysis patterns:

```yaml
feature_categories:
  mimo_optimization:
    parameters: ["antenna_configuration", "transmission_mode"]
    
  carrier_aggregation: 
    parameters: ["scell_configuration", "ca_bandwidth"]
    
  mobility_enhancement:
    parameters: ["handover_thresholds", "mobility_timers"]
    
  anr_optimization:
    parameters: ["neighbor_detection", "relation_management"]
    
  power_control:
    parameters: ["power_limits", "power_control_algorithms"]
```

## 5. Performance Management Parameters

### 5.1 Performance Counter Patterns

#### Counter Naming Convention
```
pm[Context][Action][Object][Qualifier]_[Index]
```

**Examples:**
- `pmDrbEstabSucc5qi_1` - DRB establishment success for 5QI 1
- `pmUeCategoryDistr_0` - UE category distribution index 0  
- `pmPdcpVolDlDrbQci_1` - PDCP volume downlink for QCI 1
- `pmMacHarqDlAck16Qam` - MAC HARQ DL ACK for 16QAM

#### 4G Performance Counters
| Counter Category | Pattern | Example |
|------------------|---------|---------|
| UE Category | `pmUeCategoryDistr_X` | `pmUeCategoryDistr_0` |
| PDCP Volume | `pmPdcpVolDlDrbQci_X` | `pmPdcpVolDlDrbQci_1` |
| RLC SDU | `pmRlcSduRecUlVoipTtiBundling_X` | `pmRlcSduRecUlVoipTtiBundling_0` |
| Inactivity | `pmPdcpInactSecDlVolteDistr_X` | `pmPdcpInactSecDlVolteDistr_0` |

#### 5G Performance Counters  
| Counter Category | Pattern | Example |
|------------------|---------|---------|
| DRB Establishment | `pmDrbEstab*5qi_X` | `pmDrbEstabSucc5qi_1` |
| Carrier Aggregation | `pmCaConfig*` | `pmCaConfigSucc` |
| MAC HARQ | `pmMacHarqDl*` | `pmMacHarqDlAck256Qam` |
| Abnormal Release | `pmDrbRelAbnormal*5qi_X` | `pmDrbRelAbnormalGnb5qi_1` |

## 6. Quality of Service Parameters

### 6.1 4G QCI-Based Parameters
| QCI | Service Type | Key Parameters |
|-----|--------------|----------------|
| 1 | VoLTE | `pmServiceTimeDrbQci_1` |
| 5 | IMS Signaling | `pmPdcpVolDlDrbQci_5` |
| 6 | Video | `pmPdcpVolDlDrbQci_6` |
| 65 | MCPTT | `pmPdcpVolDlDrbQci_65` |
| 66 | Non-MCPTT | `pmPdcpVolDlDrbQci_66` |

### 6.2 5G 5QI-Based Parameters
| 5QI | Service Type | Key Parameters |
|-----|--------------|----------------|
| 1 | Voice | `pmDrbEstabSucc5qi_1` |
| 5 | IMS Signaling | `pmDrbEstabSucc5qi_5` |
| 130 | FWA | `pmDrbEstabSucc5qi_130` |

## 7. Measurement and KPI Parameters

### 7.1 4G KPI Definitions
Based on `4G_KPIs.csv` analysis:

| KPI Name | Numerator Pattern | Denominator Pattern | Purpose |
|----------|-------------------|---------------------|---------|
| `%UE_CAT_X` | `pmUeCategoryDistr_X` | Sum of all distributions | UE capability analysis |
| `%_DL_VOLUME_PDCP_QCI_X` | `pmPdcpVolDlDrbQci_X` | Sum of all QCI volumes | Traffic analysis |
| `%_DL_GAPS_DURATION_IN_VOLTE_TRAFFIC` | Weighted sum formula | `pmServiceTimeDrbQci_1` | VoLTE quality |

### 7.2 5G KPI Definitions  
Based on `5G_KPIs.csv` analysis:

| KPI Name | Key Parameters | Technology |
|----------|----------------|------------|
| `5GSA_Cssr_Voice_%` | DRB accessibility chain | SA |
| `5GSA_DRB_Retainability_Active_Voice` | `pmDrbRelAbnormal*5qi_1` | SA |
| `DL_16QAM_usage` | `pmMacHarqDl*16Qam` | SA+NSA |
| `Endc_DL_CA_Configured_Scell_Ratio` | `pmCaConfigDlSumEndcDistr_*` | NSA |

## 8. Parameter Validation and Constraints

### 8.1 Data Types and Ranges
```rust
// From ParsedParameter structure
parameter_constraints: {
    data_type: ["int32", "string", "struct", "boolean"],
    range_validation: "range_and_values field",
    default_values: "system-provided defaults",
    mandatory_flags: "configuration requirements"
}
```

### 8.2 Parameter Dependencies
```yaml
dependency_patterns:
  feature_dependencies:
    - prerequisite_features_must_be_active
    - conflicting_features_must_be_inactive
    
  parameter_dependencies:  
    - range_dependencies_on_other_parameters
    - conditional_mandatory_parameters
    
  system_dependencies:
    - hardware_capability_requirements
    - software_version_requirements
```

### 8.3 Read-Only vs Configurable
| Category | Pattern | Example |
|----------|---------|---------|
| Read-Only | System identifiers | `managedElementId` |
| Read-Only | System-created | `release`, `swVersion` |
| Configurable | Operational parameters | `qRxLevMin`, `qQualMin` |
| Configurable | Feature states | `FeatureState.featureState` |

## 9. Parameter Lifecycle Management

### 9.1 Parameter States
```yaml
parameter_lifecycle:
  states:
    - preliminary: "Under development"
    - active: "In production use"
    - deprecated: "Scheduled for removal"
    - obsolete: "No longer supported"
    
  transitions:
    - preliminary → active: "Feature release"
    - active → deprecated: "Technology evolution"  
    - deprecated → obsolete: "End of life"
```

### 9.2 Change Impact Assessment
```python
def assess_parameter_change_impact(parameter: str, new_value: Any) -> ImpactAssessment:
    impacts = {
        "service_impact": check_service_disruption(parameter, new_value),
        "performance_impact": assess_kpi_effects(parameter, new_value), 
        "dependency_impact": validate_dependent_parameters(parameter),
        "rollback_complexity": estimate_rollback_effort(parameter)
    }
    return ImpactAssessment(impacts)
```

## 10. Parameter Organization for SDK

### 10.1 Proposed Parameter Repository Structure
```yaml
parameter_repository:
  categories:
    system:
      mo_classes: ["ManagedElement", "SystemFunctions"]
      parameters: ["managedElementId", "release", "swVersion"]
      
    radio_access_4g:
      mo_classes: ["EUtranCellFDD", "EUtranCellTDD"]
      parameters: ["qRxLevMin", "qQualMin", "a1a2UlSearchThreshold"]
      
    radio_access_5g:
      mo_classes: ["NRCellDU", "GNBCUCPFunction"] 
      parameters: ["pMax", "nRTAC", "pLMNIdList"]
      
    feature_management:
      mo_classes: ["FeatureState", "AdmissionControl"]
      parameters: ["featureState", "admissionControlId"]
      
    performance_monitoring:
      counter_patterns: ["pm*", "kpi_*"]
      categories: ["establishment", "volume", "quality"]
```

### 10.2 Parameter Metadata Schema
```yaml
parameter_metadata:
  identity:
    mo_class: "string"
    parameter_name: "string" 
    full_path: "string"
    
  attributes:
    data_type: "enum"
    range_values: "string"
    default_value: "any"
    unit: "string"
    
  flags:
    read_only: "boolean"
    mandatory: "boolean"
    system_created: "boolean"
    deprecated: "boolean"
    
  relationships:
    dependencies: "array"
    conflicts: "array"
    related_kpis: "array"
    
  operational:
    change_impact: "enum[low|medium|high]"
    requires_restart: "boolean"
    rollback_complexity: "enum[simple|complex]"
```

### 10.3 Parameter Search and Discovery
```python
class ParameterCatalog:
    def search_by_technology(self, tech: str) -> List[Parameter]
    def search_by_mo_class(self, mo_class: str) -> List[Parameter]  
    def search_by_function(self, function: str) -> List[Parameter]
    def get_related_parameters(self, parameter: str) -> List[Parameter]
    def validate_parameter_combination(self, params: Dict) -> ValidationResult
    def get_parameter_dependencies(self, parameter: str) -> DependencyGraph
```

## 11. Integration with Automation Workflows

### 11.1 Parameter-Workflow Mapping
```yaml
workflow_parameter_mapping:
  mobility_optimization:
    primary_parameters:
      - "EUtranCellFDD.qRxLevMin"
      - "EUtranCellFDD.a1a2UlSearchThreshold"
    monitoring_kpis:
      - "handover_success_rate"
      - "mobility_related_drops"
      
  volte_optimization:
    primary_parameters:
      - "qci_1_parameters"
      - "tti_bundling_parameters"
    monitoring_counters:
      - "pmPdcpInactSecDlVolteDistr_*"
      - "pmRlcSduRecUlVoipTtiBundling_*"
```

### 11.2 Parameter Validation Integration
```python
def validate_workflow_parameters(workflow: str, parameters: Dict) -> ValidationResult:
    validations = [
        validate_parameter_ranges(parameters),
        check_technology_compatibility(workflow, parameters),
        verify_feature_dependencies(parameters),
        assess_service_impact(parameters)
    ]
    return combine_validation_results(validations)
```

## 12. Recommendations for SDK Parameter Management

### 12.1 Core Features
1. **Comprehensive Catalog**: Complete parameter repository with metadata
2. **Smart Search**: Technology, function, and pattern-based discovery
3. **Dependency Management**: Automatic dependency resolution and validation
4. **Impact Assessment**: Pre-change impact analysis
5. **Version Management**: Parameter lifecycle and deprecation tracking

### 12.2 Advanced Capabilities
1. **AI-Powered Recommendations**: ML-based parameter optimization suggestions
2. **Conflict Detection**: Automatic parameter conflict identification
3. **Template Management**: Pre-configured parameter sets for common scenarios
4. **Rollback Planning**: Automatic rollback plan generation
5. **Compliance Checking**: Regulatory and best practice validation

### 12.3 Integration Points
1. **CMEDIT Integration**: Direct CLI command generation
2. **KPI Integration**: Parameter-KPI relationship mapping
3. **Workflow Integration**: Seamless workflow parameter binding
4. **Documentation Generation**: Automatic parameter documentation
5. **Testing Support**: Parameter validation test generation

This parameter catalog provides the foundation for building comprehensive parameter management capabilities in the RAN automation SDK.