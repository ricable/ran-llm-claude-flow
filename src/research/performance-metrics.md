# RAN Performance Metrics Research Report

## Executive Summary

This document provides a comprehensive analysis of Ericsson RAN performance metrics, KPIs, and counters identified from the codebase to inform SDK performance monitoring and optimization capabilities.

## 1. Performance Metrics Architecture

### 1.1 Metrics Hierarchy
The codebase reveals a structured approach to RAN performance monitoring:

```
Performance Metrics
├── Raw Counters (pm* patterns)
├── Derived KPIs (calculated metrics)
├── Service-Level Metrics (user experience)
└── Network-Level Metrics (overall performance)
```

### 1.2 Technology Classification
Performance metrics are organized by technology:

- **4G/LTE Metrics**: EUtran-based counters and KPIs
- **5G/NR Metrics**: NR-specific counters and KPIs  
- **Multi-RAT Metrics**: Cross-technology performance indicators
- **Transport Metrics**: Backhaul and transport performance

## 2. Performance Counter Patterns

### 2.1 Counter Naming Convention
Based on extensive analysis of pm* patterns:

```
pm[Context][Action][Object][Qualifier]_[Index]
```

**Pattern Breakdown:**
- `pm`: Performance Management prefix
- `[Context]`: Radio layer context (Drb, Pdcp, Rlc, Mac, etc.)
- `[Action]`: Operation type (Estab, Vol, Harq, etc.)  
- `[Object]`: Target object (Ul, Dl, QCI, 5QI, etc.)
- `[Qualifier]`: Success/failure, direction, type
- `[Index]`: Numerical index or identifier

### 2.2 4G/LTE Performance Counters

#### DRB (Data Radio Bearer) Counters
```yaml
drb_establishment:
  pattern: "pmDrbEstab*"
  examples:
    - pmDrbEstabAtt: "DRB establishment attempts"
    - pmDrbEstabSucc: "DRB establishment successes" 
    - pmDrbEstabFail: "DRB establishment failures"
  
drb_release:
  pattern: "pmDrbRel*"  
  examples:
    - pmDrbRelNormal: "Normal DRB releases"
    - pmDrbRelAbnormal: "Abnormal DRB releases"
```

#### PDCP (Packet Data Convergence Protocol) Counters
```yaml
pdcp_volume:
  pattern: "pmPdcpVol*"
  examples:
    - pmPdcpVolDlDrbQci_1: "PDCP DL volume for QCI 1"
    - pmPdcpVolUlDrbQci_1: "PDCP UL volume for QCI 1"
    - pmPdcpVolDlDrbQci_65: "PDCP DL volume for MCPTT"
    
pdcp_packets:
  pattern: "pmPdcpPkt*"
  examples:
    - pmPdcpPktReceivedUlQci_1: "PDCP packets received UL for QCI 1"
    - pmPdcpPktLostUlQci_1: "PDCP packets lost UL for QCI 1"
    
pdcp_inactivity:
  pattern: "pmPdcpInactSec*"  
  examples:
    - pmPdcpInactSecDlVolteDistr_0: "DL inactivity distribution for VoLTE"
    - pmPdcpInactSecDlmcpttDistr_0: "DL inactivity distribution for MCPTT"
```

#### MAC (Medium Access Control) Counters
```yaml
mac_harq:
  pattern: "pmMacHarq*"
  examples:
    - pmMacHarqDlAckqpsk: "MAC HARQ DL ACK for QPSK"
    - pmMacHarqDlAck16Qam: "MAC HARQ DL ACK for 16QAM"
    - pmMacHarqDlAck64Qam: "MAC HARQ DL ACK for 64QAM"
    - pmMacHarqDlAck256Qam: "MAC HARQ DL ACK for 256QAM"
```

#### RLC (Radio Link Control) Counters  
```yaml
rlc_sdu:
  pattern: "pmRlcSdu*"
  examples:
    - pmRlcSduRecUlVoipTtiBundling_0: "RLC SDU received UL VoIP TTI bundling"
    - pmRlcSduRecUlmcpttTtiBundl_0: "RLC SDU received UL MCPTT TTI bundling"
    - pmRlcSduEstLostUlVoipTtiBundling_0: "RLC SDU lost UL VoIP TTI bundling"
```

#### UE Category Distribution
```yaml
ue_category:
  pattern: "pmUeCategoryDistr_*"
  examples:
    - pmUeCategoryDistr_0: "UE Category 1 distribution"
    - pmUeCategoryDistr_1: "UE Category 2 distribution"
    - pmUeCategoryDistr_9: "UE Category 10 distribution"
  
ue_category_combined:
  pattern: "pmUeCategoryDlUlCombDistr_*"
  examples:
    - pmUeCategoryDlUlCombDistr_0: "Combined DL/UL category distribution"
```

### 2.3 5G/NR Performance Counters

#### DRB Counters (5QI-based)
```yaml
drb_establishment_5g:
  pattern: "pmDrbEstab*5qi_*"
  examples:
    - pmDrbEstabAtt5qi_1: "DRB establishment attempts for 5QI 1 (Voice)"
    - pmDrbEstabSucc5qi_1: "DRB establishment success for 5QI 1 (Voice)"
    - pmDrbEstabSucc5qi_5: "DRB establishment success for 5QI 5 (IMS)"
    - pmDrbEstabSucc5qi_130: "DRB establishment success for 5QI 130 (FWA)"

drb_release_5g:
  pattern: "pmDrbRel*5qi_*"
  examples:
    - pmDrbRelAbnormalGnb5qi_1: "Abnormal DRB release by gNB for 5QI 1"
    - pmDrbRelAbnormalAmf5qi_1: "Abnormal DRB release by AMF for 5QI 1"
    - pmDrbRelnormal5qi_5: "Normal DRB release for 5QI 5"
```

#### Carrier Aggregation Counters
```yaml
carrier_aggregation:
  pattern: "pmCa*"
  examples:
    - pmCaConfigAtt: "CA configuration attempts"
    - pmCaConfigSucc: "CA configuration successes"
    - pmCaConfigDlSumEndcDistr_1: "CA DL configuration distribution for EN-DC"
```

#### PDCP Counters (5QI-based)
```yaml
pdcp_5g:
  pattern: "pmPdcpPkt*5qi_*"
  examples:
    - pmPdcpPktRecDl5qi_1: "PDCP packets received DL for 5QI 1"
    - pmPdcpPktRecDlDiscAqm5qi_1: "PDCP packets discarded due to AQM for 5QI 1"
```

#### MAC HARQ (5G Modulation)
```yaml
mac_harq_5g:
  pattern: "pmMacHarqDl*"
  examples:
    - pmMacHarqDlAck256Qam: "MAC HARQ DL ACK for 256QAM"
    - pmMacHarqDlNack256qam: "MAC HARQ DL NACK for 256QAM"
```

## 3. Key Performance Indicators (KPIs)

### 3.1 4G/LTE KPIs

#### UE Category KPIs
From `4G_KPIs.csv` analysis:

```yaml
ue_category_kpis:
  "%UE_CAT_1":
    formula: "100*pmUeCategoryDistr_0 / (sum of all category distributions)"
    purpose: "UE Category 1 percentage distribution"
    
  "%UE_CAT_4": 
    formula: "100*pmUeCategoryDistr_3 / (sum of all category distributions)"
    purpose: "UE Category 4 percentage distribution"
    status: "In Production"
    
  "%UE_CAT_7":
    formula: "100*pmUeCategoryDistr_6 / (sum of all category distributions)" 
    purpose: "UE Category 7 percentage distribution"
    status: "In Production"
```

#### Volume Distribution KPIs
```yaml
volume_kpis:
  "%_DL_VOLUME_PDCP_QCI_1":
    formula: "100*pmPdcpVolDlDrbQci_1 / (sum of all QCI volumes)"
    purpose: "VoLTE traffic percentage"
    priority: "Primaire"
    
  "%_DL_VOLUME_PDCP_QCI_5":
    formula: "100*pmPdcpVolDlDrbQci_5 / (sum of all QCI volumes)"
    purpose: "IMS signaling traffic percentage"
    
  "%_DL_VOLUME_PDCP_QCI_65":
    formula: "100*pmPdcpVolDlDrbQci_65 / (sum of all QCI volumes)"
    purpose: "MCPTT traffic percentage"
```

#### VoLTE Quality KPIs
```yaml
volte_quality_kpis:
  "%_DL_GAPS_DURATION_IN_VOLTE_TRAFFIC":
    formula: "100*(weighted sum of pmPdcpInactSecDlVolteDistr_*) / pmServiceTimeDrbQci_1"
    purpose: "DL silence duration in VoLTE calls"
    priority: "Secondaire"
    
  "%_PDU_VOLTE_EN_TTI_BUNDLING":
    formula: "100*(sum of pmRlcSduRecUlVoipTtiBundling_* + pmRlcSduEstLostUlVoipTtiBundling_*) / (pmPdcpPktLostUlQci_1 + pmPdcpPktReceivedUlQci_1)"
    purpose: "VoLTE PDU ratio using TTI bundling"
```

#### MCPTT KPIs
```yaml
mcptt_kpis:
  "%_Pdu_MCPTT_en_TTI_Bundling":
    formula: "Complex sum of pmRlcSduRecUlmcpttTtiBundl_* counters"
    purpose: "MCPTT PDU ratio using TTI bundling"
    
  "%_Dl_Gaps_Duration_in_MCPTT_Traffic":
    formula: "100*(weighted sum of pmPdcpInactSecDlmcpttDistr_*) / pmServiceTimeDrbQci_65"
    purpose: "DL gaps duration in MCPTT traffic"
```

### 3.2 5G/NR KPIs  

#### Accessibility KPIs
From `5G_KPIs.csv` analysis:

```yaml
accessibility_5g:
  "5GSA_Cssr_Voice_%":
    formula: "[NRSA_RRC_Conn_Estab_SR]*[NRSA_NG_Sig_Conn_Estab_SR]*[5GSA_DRB_Establishment_SR_for_Voice]*[5GSA_DRB_Establishment_SR_for_IMS_Signaling]"
    purpose: "DRB Accessibility Success Rate for Voice Traffic"
    priority: "Primaire"
    technology: "SA+NSA"
    
  "5GSA_DRB_Estab_SR":
    formula: "100*pmDrbEstabSucc / pmDrbEstabAtt"
    purpose: "Overall DRB establishment success rate"
    
  "5GSA_Drb_Estab_SR_5qi_130":
    formula: "100*pmDrbEstabSucc5qi_130 / pmDrbEstabAtt5qi_130"  
    purpose: "DRB establishment success rate for 5QI 130 (FWA)"
```

#### Retainability KPIs
```yaml
retainability_5g:
  "5GSA_DRB_Retainability_Active_Voice":
    formula: "100*(pmDrbRelAbnormalGnbAct5qi_1 + pmDrbRelAbnormalAmfAct5qi_1) / (pmDrbRelnormal5qi_1 + pmDrbRelAbnormalAmf5qi_1 + pmDrbRelAbnormalGnb5qi_1)"
    purpose: "DRB Retainability - Percentage of Active Lost for Voice Traffic"
    priority: "Primaire"
    
  "5GSA_Drop_Rate_5qi_5":
    formula: "100*(pmDrbRelAbnormalGnb5qi_5 + pmDrbRelAbnormalAmf5qi_5) / (pmDrbRelnormal5qi_5 + pmDrbRelAbnormalAmf5qi_5 + pmDrbRelAbnormalGnb5qi_5)"
    purpose: "Drop rate for IMS signaling"
```

#### Carrier Aggregation KPIs
```yaml
carrier_aggregation_5g:
  "5GSA_DL_CA_Config_SR":
    formula: "100*pmCaConfigSucc / pmCaConfigAtt"
    purpose: "SCell configuration success rate"
    priority: "Primaire"
    technology: "SA"
    
  "Endc_DL_CA_Configured_Scell_Ratio":  
    formula: "100*(sum of pmCaConfigDlSumEndcDistr_1 to _7) / pmCaConfigDlSumEndcDistr_SUM"
    purpose: "Percentage of EN-DC UEs with configured SCells"
    priority: "Primaire"
    technology: "NSA"
```

#### Modulation Usage KPIs
```yaml
modulation_kpis:
  "DL_16QAM_usage":
    formula: "100*(pmMacHarqDlAck16Qam + pmMacHarqDlNack16Qam) / (sum of all modulation ACK/NACK)"
    purpose: "16QAM modulation usage ratio"
    priority: "Primaire"
    
  "DL_256QAM_usage":
    formula: "100*(pmMacHarqDlAck256Qam + pmMacHarqDlNack256Qam) / (sum of all modulation ACK/NACK)"
    purpose: "256QAM modulation usage ratio"
    priority: "Primaire"
```

#### Quality KPIs
```yaml
quality_5g:
  "5GSA_DL_Packet_Loss_Rate_AQM_Voice":
    formula: "100*pmPdcpPktRecDlDiscAqm5qi_1 / pmPdcpPktRecDl5qi_1"
    purpose: "DL packet loss rate due to AQM for voice"
    technology: "SA"
```

### 3.3 FWA (Fixed Wireless Access) KPIs
```yaml
fwa_kpis:
  "%_de_trafic_FWA_en_SA":
    formula: "100*[Flex_DL_Vol_Qos130_SA] / ([Flex_DL_Vol_Qos130_SA]+[Flex_DL_Vol_Qos130_NSA])"
    purpose: "FWA traffic percentage in Stand Alone mode"
    qos: "QoS 130"
```

## 4. Transport and Interface Metrics

### 4.1 Ethernet Interface Metrics
```yaml
transport_kpis:
  "100%-ile_Egress_Usage_Avg":
    formula: "8*ifOutOctetRatePercentiles_6 / 1000000"
    unit: "Mbits/s"
    purpose: "Ethernet interface average egress usage"
    
  "100%-ile_Ingress_Usage_Avg":
    formula: "8*ifInOctetRatePercentiles_6 / 1000000" 
    unit: "Mbits/s"
    purpose: "Ethernet interface average ingress usage"
```

## 5. Service-Specific Metrics

### 5.1 VoNR (Voice over NR) Metrics
```yaml
vonr_metrics:
  establishment:
    - "5GSA_Estab_Succ_5qi_1": "Voice DRB establishment successes"
    - "5GSA_Estab_SR_5qi_1": "Voice DRB establishment success rate"
    - "5GSA_Estab_SR_5qi_5": "IMS signaling establishment success rate"
    
  retainability:
    - "5GSA_Abnormal_Drop_Voice": "Voice call abnormal drops"
    - "5GSA_Abnormal_Drop_Voice_gNB": "gNB-triggered voice drops"
    - "5GSA_Abnormal_Drop_Voice_Amf": "AMF-triggered voice drops"
```

### 5.2 MCPTT (Mission Critical Push-to-Talk) Metrics
```yaml
mcptt_metrics:
  traffic_distribution:
    - "%_Dl_Gaps_Duration_in_MCPTT_Traffic": "Downlink silence analysis"
    - "%_Pdu_MCPTT_en_TTI_Bundling": "TTI bundling efficiency"
    
  quality_indicators:
    - "mcptt_latency_metrics": "End-to-end latency measurements"
    - "mcptt_reliability_metrics": "Service reliability indicators"
```

## 6. Advanced Performance Analytics

### 6.1 Mobility Performance Metrics
Based on mobility trigger analysis:

```yaml
mobility_metrics:
  handover_success:
    formula: "calculated from A1/A2/A3/A4/A5 event counters"
    triggers: "Based on RSRP/RSRQ thresholds"
    
  cell_reselection:
    formula: "Based on qRxLevMin/qQualMin criteria"
    idle_mode: "Cell selection criteria evaluation"
    
  inter_frequency_mobility:
    formula: "Cross-frequency handover metrics"
    dependencies: "Frequency priority settings"
```

### 6.2 Load and Capacity Metrics
```yaml
capacity_metrics:
  resource_utilization:
    - "prb_utilization_dl": "Physical Resource Block usage DL"
    - "prb_utilization_ul": "Physical Resource Block usage UL"
    
  user_throughput:
    - "average_user_throughput_dl": "Per-user DL throughput"
    - "cell_throughput_capacity": "Cell-level capacity metrics"
```

## 7. Performance Monitoring Workflows

### 7.1 Real-Time Monitoring Pattern
```python
def real_time_monitoring_workflow(cell_dn: str, duration_minutes: int):
    """
    Real-time performance monitoring workflow
    """
    counters_to_monitor = [
        "pmDrbEstabAtt", "pmDrbEstabSucc",
        "pmHandoverSucc", "pmHandoverAtt", 
        "pmPdcpVolDlDrbQci_1", "pmMacHarqDlAck*"
    ]
    
    workflow = [
        f"get {cell_dn} {','.join(counters_to_monitor)} -interval=15s",
        f"calculate derived_kpis -real-time",
        f"monitor thresholds -duration={duration_minutes}m",
        f"alert on_threshold_breach -severity=major"
    ]
    
    return workflow
```

### 7.2 KPI Trending Analysis
```python
def kpi_trending_analysis(cell_collection: str, timeframe: str):
    """
    Historical KPI trend analysis workflow
    """
    
    key_kpis = [
        "DRB_Establishment_SR", "Call_Drop_Rate", 
        "Average_Throughput", "Handover_Success_Rate"
    ]
    
    workflow = [
        f"collect_historical_counters collection={cell_collection} timeframe={timeframe}",
        f"calculate_kpis kpis={','.join(key_kpis)}",
        f"analyze_trends -statistical-analysis",
        f"identify_anomalies -threshold-deviation=2-sigma", 
        f"generate_optimization_recommendations"
    ]
    
    return workflow
```

## 8. Performance Optimization Patterns

### 8.1 Threshold-Based Optimization
```yaml
optimization_thresholds:
  drb_establishment_sr:
    warning: 95.0
    critical: 90.0
    action: "investigate_signaling_issues"
    
  call_drop_rate:
    warning: 2.0  
    critical: 5.0
    action: "optimize_mobility_parameters"
    
  average_throughput:
    warning: "10% below baseline"
    critical: "20% below baseline" 
    action: "capacity_expansion_analysis"
```

### 8.2 AI-Driven Performance Optimization
```python
def ai_performance_optimization(performance_data: Dict) -> OptimizationPlan:
    """
    Use AI/ML models to optimize RAN performance
    """
    
    optimization_areas = [
        "mobility_parameter_tuning",
        "load_balancing_optimization", 
        "interference_mitigation",
        "capacity_planning"
    ]
    
    # AI analysis of performance patterns
    insights = ai_engine.analyze_performance(performance_data)
    
    # Generate optimization recommendations
    recommendations = ai_engine.generate_recommendations(insights)
    
    # Create executable optimization plan
    plan = create_optimization_plan(recommendations)
    
    return plan
```

## 9. Performance Reporting Framework

### 9.1 Standard Performance Reports
```yaml
performance_reports:
  daily_summary:
    kpis: ["accessibility", "retainability", "throughput", "quality"]
    format: "executive_summary"
    delivery: "automated_email"
    
  weekly_trend_analysis:
    kpis: ["all_key_metrics"]
    analysis: ["trend_detection", "anomaly_identification"]
    format: "detailed_charts_and_analysis"
    
  monthly_optimization_report:
    content: ["performance_baseline", "optimization_opportunities", "action_plan"]
    stakeholders: ["network_planning", "optimization_team"]
```

### 9.2 Custom Report Generation
```python  
class PerformanceReportGenerator:
    
    def generate_custom_report(
        self, 
        metrics: List[str],
        timeframe: str,
        aggregation_level: str,
        format: str
    ) -> PerformanceReport:
        
        # Collect raw data
        raw_data = self.collect_performance_data(metrics, timeframe)
        
        # Apply aggregation
        aggregated_data = self.aggregate_by_level(raw_data, aggregation_level)
        
        # Calculate derived KPIs
        kpis = self.calculate_derived_kpis(aggregated_data)
        
        # Generate insights
        insights = self.analyze_performance_trends(kpis)
        
        # Format report
        report = self.format_report(kpis, insights, format)
        
        return report
```

## 10. Integration with SDK Architecture

### 10.1 Performance Monitoring API
```python
class RanPerformanceMonitor:
    
    def collect_counters(self, node_dn: str, counter_list: List[str]) -> CounterData:
        """Collect performance counters from RAN nodes"""
        
    def calculate_kpi(self, kpi_name: str, timeframe: str) -> KpiValue:
        """Calculate derived KPI from raw counters"""
        
    def monitor_real_time(self, metrics: List[str], callback: Callable) -> MonitorSession:
        """Start real-time performance monitoring"""
        
    def analyze_trends(self, metrics: List[str], period: str) -> TrendAnalysis:
        """Perform historical trend analysis"""
        
    def detect_anomalies(self, baseline_data: Dict, current_data: Dict) -> AnomalyReport:
        """Detect performance anomalies"""
```

### 10.2 Performance Optimization Engine
```python
class PerformanceOptimizationEngine:
    
    def assess_current_performance(self, scope: str) -> PerformanceAssessment:
        """Assess current network performance"""
        
    def identify_optimization_opportunities(self, assessment: PerformanceAssessment) -> List[OptimizationOpportunity]:
        """Identify areas for performance improvement"""
        
    def generate_optimization_plan(self, opportunities: List[OptimizationOpportunity]) -> OptimizationPlan:
        """Create executable optimization plan"""
        
    def execute_optimization(self, plan: OptimizationPlan) -> OptimizationResult:
        """Execute performance optimization actions"""
        
    def validate_optimization_results(self, baseline: Dict, post_optimization: Dict) -> ValidationResult:
        """Validate optimization effectiveness"""
```

## 11. Recommendations for SDK Implementation

### 11.1 Performance Monitoring Framework
1. **Real-Time Monitoring**: Sub-second performance metric collection and alerting
2. **Historical Analysis**: Long-term trend analysis and baseline management
3. **Predictive Analytics**: AI-powered performance forecasting and proactive optimization
4. **Multi-Technology Support**: Unified monitoring across 4G, 5G, and multi-RAT scenarios
5. **Scalable Architecture**: Support for network-wide performance monitoring

### 11.2 KPI Management System
1. **KPI Catalog**: Comprehensive library of standard and custom KPIs
2. **Formula Engine**: Flexible KPI calculation with configurable formulas
3. **Threshold Management**: Dynamic threshold setting with statistical baselines
4. **Visualization**: Rich charting and dashboard capabilities
5. **Reporting Engine**: Automated report generation and distribution

### 11.3 Optimization Capabilities
1. **AI-Driven Optimization**: Machine learning-based parameter optimization
2. **What-If Analysis**: Performance impact modeling before changes
3. **Automated Tuning**: Self-optimizing network capabilities
4. **Rollback Protection**: Safe optimization with automatic rollback
5. **Best Practices Integration**: Built-in industry best practices and compliance

This performance metrics analysis provides the blueprint for implementing comprehensive RAN performance monitoring, analysis, and optimization capabilities in the SDK architecture.