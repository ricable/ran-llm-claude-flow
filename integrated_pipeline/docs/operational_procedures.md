# Operational Procedures - Production Hybrid Pipeline

## 🎯 **Enterprise Operational Procedures Guide**

**Environment**: Production M3 Max Deployment  
**Pipeline**: Hybrid Rust-Python with Advanced Features  
**Version**: Weeks 5-8 Implementation  
**Target SLA**: 99% uptime, 35+ docs/hour, <2s response time  

---

## 📋 **Daily Operations Procedures**

### Morning Startup Routine (8:00 AM)
```bash
#!/bin/bash
# daily_startup_check.sh

echo "=== Daily Production Health Check ==="
echo "Date: $(date)"

# 1. System Health Verification
echo "1. Checking system health..."
./scripts/health_check.sh --comprehensive

# 2. Performance Metrics Review
echo "2. Reviewing overnight performance..."
python scripts/performance_summary.py --period=24h

# 3. Error Log Analysis
echo "3. Analyzing error logs..."
./scripts/error_analysis.sh --since="24 hours ago"

# 4. Resource Utilization Check
echo "4. Checking resource utilization..."
./scripts/resource_check.sh --alert-threshold=80

# 5. Model Performance Validation
echo "5. Validating model performance..."
python scripts/model_validation.py --quick-check

echo "=== Daily Check Complete ==="
```

### Hourly Monitoring Tasks
```bash
#!/bin/bash
# hourly_monitoring.sh (runs via cron)

# Memory usage check
MEMORY_USAGE=$(python -c "
import psutil
memory = psutil.virtual_memory()
print(f'{memory.percent:.1f}')
")

if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "HIGH MEMORY USAGE: ${MEMORY_USAGE}%" | logger -t hybrid-pipeline
    ./scripts/memory_cleanup.sh
fi

# Temperature monitoring (M3 Max specific)
TEMP=$(sudo powermetrics -s thermal -n 1 | grep "CPU die temperature" | awk '{print $4}')
if (( $(echo "$TEMP > 85" | bc -l) )); then
    echo "HIGH TEMPERATURE: ${TEMP}°C" | logger -t hybrid-pipeline
    ./scripts/thermal_management.sh
fi

# Throughput validation
python scripts/throughput_check.py --threshold=30
```

### Evening Maintenance (6:00 PM)
```bash
#!/bin/bash
# evening_maintenance.sh

echo "=== Evening Maintenance Routine ==="

# 1. Performance optimization
echo "1. Running performance optimization..."
python scripts/performance_optimizer.py --auto-tune

# 2. Log rotation and cleanup
echo "2. Rotating logs..."
./scripts/log_rotation.sh

# 3. Metrics aggregation
echo "3. Aggregating daily metrics..."
python scripts/metrics_aggregation.py --period=daily

# 4. Backup validation
echo "4. Validating today's backups..."
./scripts/backup_validation.sh

# 5. Predictive analysis
echo "5. Running predictive analysis..."
python scripts/predictive_analysis.py --forecast=24h

echo "=== Evening Maintenance Complete ==="
```

---

## 📊 **Monitoring and Alerting Procedures**

### Critical Alert Response (Severity: Critical)
```bash
#!/bin/bash
# critical_alert_response.sh

ALERT_TYPE=$1
COMPONENT=$2

echo "CRITICAL ALERT: $ALERT_TYPE in $COMPONENT"

case $ALERT_TYPE in
    "HIGH_MEMORY_USAGE")
        echo "Responding to high memory usage..."
        # 1. Immediate memory analysis
        python scripts/memory_analyzer.py --detailed
        
        # 2. Emergency memory cleanup
        ./scripts/emergency_memory_cleanup.sh
        
        # 3. Consider service restart if >95%
        MEMORY_PCT=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
        if (( $(echo "$MEMORY_PCT > 95" | bc -l) )); then
            ./scripts/graceful_restart.sh --component=high-memory-consumers
        fi
        ;;
        
    "HIGH_ERROR_RATE")
        echo "Responding to high error rate..."
        # 1. Error pattern analysis
        ./scripts/error_pattern_analysis.sh --last-hour
        
        # 2. Service health check
        ./scripts/service_health_check.sh --all
        
        # 3. Potential rollback if recent deployment
        ./scripts/check_recent_deployment.sh --auto-rollback
        ;;
        
    "PERFORMANCE_DEGRADATION")
        echo "Responding to performance degradation..."
        # 1. Bottleneck identification
        python scripts/bottleneck_analyzer.py --real-time
        
        # 2. Resource reallocation
        ./scripts/resource_reallocation.sh --optimize
        
        # 3. Model performance check
        python scripts/model_performance_check.py --all-models
        ;;
esac

# Always notify team
./scripts/send_alert_notification.sh --severity=critical --type=$ALERT_TYPE
```

### Warning Alert Response (Severity: Warning)
```bash
#!/bin/bash
# warning_alert_response.sh

ALERT_TYPE=$1

case $ALERT_TYPE in
    "QUALITY_DEGRADATION")
        # Investigate model quality issues
        python scripts/quality_analysis.py --detailed
        python scripts/model_comparison.py --recent-performance
        ;;
        
    "THROUGHPUT_DECLINE")
        # Analyze processing pipeline
        python scripts/pipeline_analysis.py --bottlenecks
        ./scripts/throughput_optimization.sh
        ;;
        
    "TEMPERATURE_WARNING")
        # Thermal management
        ./scripts/thermal_analysis.sh
        sudo pmset -a perfbias 1  # Slightly reduce performance for cooling
        ;;
esac

# Log warning and continue monitoring
echo "$(date): Warning alert $ALERT_TYPE handled" >> /opt/hybrid-pipeline/logs/alerts.log
```

---

## 🔧 **Maintenance Procedures**

### Weekly System Optimization
```bash
#!/bin/bash
# weekly_optimization.sh (runs every Sunday)

echo "=== Weekly System Optimization ==="

# 1. Deep performance analysis
echo "1. Running deep performance analysis..."
python scripts/deep_performance_analysis.py --week

# 2. Model retraining evaluation
echo "2. Evaluating model retraining needs..."
python scripts/model_retraining_evaluator.py

# 3. Database maintenance
echo "3. Performing database maintenance..."
psql hybrid_pipeline_production -c "VACUUM ANALYZE;"
psql hybrid_pipeline_production -c "REINDEX DATABASE hybrid_pipeline_production;"

# 4. Cache optimization
echo "4. Optimizing cache performance..."
redis-cli FLUSHDB  # Clear cache for fresh start
python scripts/cache_warmup.py --comprehensive

# 5. Configuration optimization
echo "5. Reviewing configuration optimization..."
python scripts/config_optimizer.py --suggest-improvements

# 6. Security updates
echo "6. Applying security updates..."
./scripts/security_updates.sh --auto-approve-critical

# 7. Performance benchmarking
echo "7. Running performance benchmarks..."
./scripts/benchmark_suite.sh --comprehensive

echo "=== Weekly Optimization Complete ==="
```

### Monthly Deep Maintenance
```bash
#!/bin/bash
# monthly_maintenance.sh (runs first Sunday of month)

echo "=== Monthly Deep Maintenance ==="

# 1. Comprehensive system audit
echo "1. System audit..."
./scripts/system_audit.sh --detailed

# 2. Capacity planning analysis
echo "2. Capacity planning..."
python scripts/capacity_planning.py --forecast=3months

# 3. Long-term performance trends
echo "3. Performance trend analysis..."
python scripts/performance_trends.py --period=30days

# 4. Model performance evaluation
echo "4. Model performance evaluation..."
python scripts/model_evaluation_suite.py --comprehensive

# 5. Infrastructure optimization
echo "5. Infrastructure optimization..."
./scripts/infrastructure_optimizer.sh --deep-analysis

# 6. Disaster recovery testing
echo "6. DR testing..."
./scripts/disaster_recovery_test.sh --non-destructive

echo "=== Monthly Maintenance Complete ==="
```

---

## 🚨 **Incident Response Procedures**

### Service Outage Response
```bash
#!/bin/bash
# incident_response.sh

INCIDENT_TYPE=$1
SEVERITY=$2

echo "=== INCIDENT RESPONSE: $INCIDENT_TYPE (Severity: $SEVERITY) ==="

# 1. Immediate assessment
echo "1. Assessing system status..."
./scripts/system_status_check.sh --emergency

# 2. Start incident log
INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"
echo "Incident ID: $INCIDENT_ID" > /opt/hybrid-pipeline/logs/incident_$INCIDENT_ID.log

# 3. Service isolation
echo "2. Isolating affected services..."
case $INCIDENT_TYPE in
    "RUST_CORE_FAILURE")
        systemctl stop rust-core-service
        ./scripts/rust_core_diagnostics.sh
        ;;
    "PYTHON_ML_FAILURE") 
        systemctl stop python-ml-service
        ./scripts/python_ml_diagnostics.sh
        ;;
    "IPC_FAILURE")
        ./scripts/ipc_reset.sh
        ;;
esac

# 4. Attempt automatic recovery
echo "3. Attempting automatic recovery..."
./scripts/auto_recovery.sh --incident-type=$INCIDENT_TYPE

# 5. Manual intervention if needed
if [ $? -ne 0 ]; then
    echo "4. Automatic recovery failed, initiating manual procedures..."
    ./scripts/manual_recovery.sh --incident-id=$INCIDENT_ID
fi

# 6. Service validation
echo "5. Validating service recovery..."
./scripts/service_validation.sh --post-incident

# 7. Incident documentation
echo "6. Documenting incident..."
python scripts/incident_documentation.py --incident-id=$INCIDENT_ID

echo "=== INCIDENT RESPONSE COMPLETE ==="
```

### Recovery Validation
```bash
#!/bin/bash
# recovery_validation.sh

echo "=== Recovery Validation ==="

# 1. Service health checks
echo "1. Checking service health..."
for service in rust-core python-ml ipc-layer monitoring; do
    if systemctl is-active $service >/dev/null 2>&1; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
        exit 1
    fi
done

# 2. Performance validation
echo "2. Validating performance..."
THROUGHPUT=$(python scripts/quick_throughput_test.py)
if (( $(echo "$THROUGHPUT > 25" | bc -l) )); then
    echo "✅ Throughput acceptable: $THROUGHPUT docs/hour"
else
    echo "❌ Throughput below threshold: $THROUGHPUT docs/hour"
    exit 1
fi

# 3. Quality validation
echo "3. Validating quality..."
QUALITY=$(python scripts/quick_quality_test.py)
if (( $(echo "$QUALITY > 0.75" | bc -l) )); then
    echo "✅ Quality acceptable: $QUALITY"
else
    echo "❌ Quality below threshold: $QUALITY"
    exit 1
fi

# 4. End-to-end test
echo "4. Running end-to-end test..."
python scripts/e2e_test.py --quick

echo "✅ Recovery validation complete"
```

---

## 📈 **Performance Management Procedures**

### Performance Baseline Establishment
```bash
#!/bin/bash
# establish_baseline.sh

echo "=== Establishing Performance Baseline ==="

# 1. Run comprehensive benchmarks
./scripts/comprehensive_benchmark.sh --duration=1hour

# 2. Document baseline metrics
python scripts/baseline_documentation.py --save-baseline

# 3. Set monitoring thresholds
python scripts/set_monitoring_thresholds.py --based-on-baseline

# 4. Configure alerts
./scripts/configure_performance_alerts.sh --baseline-derived

echo "Performance baseline established and saved"
```

### Performance Degradation Investigation
```bash
#!/bin/bash
# investigate_performance.sh

echo "=== Performance Degradation Investigation ==="

# 1. Compare current vs baseline
python scripts/performance_comparison.py --baseline --current

# 2. Identify bottlenecks
python scripts/bottleneck_identification.py --detailed

# 3. Resource analysis
./scripts/resource_analysis.sh --historical-comparison

# 4. Model performance analysis
python scripts/model_performance_analysis.py --degradation-analysis

# 5. Generate optimization recommendations
python scripts/optimization_recommendations.py --auto-generate

echo "Performance investigation complete"
```

---

## 🔄 **Backup and Recovery Procedures**

### Daily Backup Procedure
```bash
#!/bin/bash
# daily_backup.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/opt/hybrid-pipeline/backups/daily/$BACKUP_DATE"

echo "=== Daily Backup: $BACKUP_DATE ==="

# 1. Create backup directory
mkdir -p $BACKUP_DIR

# 2. Database backup
echo "1. Backing up database..."
pg_dump hybrid_pipeline_production > $BACKUP_DIR/database.sql

# 3. Configuration backup
echo "2. Backing up configuration..."
tar -czf $BACKUP_DIR/config.tar.gz /opt/hybrid-pipeline/config/

# 4. Models backup
echo "3. Backing up models..."
rsync -av /opt/hybrid-pipeline/models/ $BACKUP_DIR/models/

# 5. Application state backup
echo "4. Backing up application state..."
python scripts/backup_application_state.py --output=$BACKUP_DIR/app_state.json

# 6. Backup verification
echo "5. Verifying backup..."
./scripts/backup_verification.sh $BACKUP_DIR

# 7. Cleanup old backups (keep 30 days)
find /opt/hybrid-pipeline/backups/daily/ -type d -mtime +30 -exec rm -rf {} \;

echo "Daily backup complete: $BACKUP_DIR"
```

### Recovery Testing Procedure
```bash
#!/bin/bash
# recovery_test.sh

TEST_ENV="/opt/hybrid-pipeline/test-recovery"
BACKUP_PATH=$1

echo "=== Recovery Test from $BACKUP_PATH ==="

# 1. Create isolated test environment
echo "1. Setting up test environment..."
mkdir -p $TEST_ENV
cp -r $BACKUP_PATH/* $TEST_ENV/

# 2. Database recovery test
echo "2. Testing database recovery..."
createdb test_recovery_db
psql test_recovery_db < $TEST_ENV/database.sql

# 3. Configuration recovery test
echo "3. Testing configuration recovery..."
tar -xzf $TEST_ENV/config.tar.gz -C $TEST_ENV/

# 4. Application startup test
echo "4. Testing application startup..."
cd $TEST_ENV && ./scripts/test_startup.sh

# 5. Functionality test
echo "5. Testing core functionality..."
python scripts/functionality_test.py --test-env=$TEST_ENV

# 6. Cleanup test environment
echo "6. Cleaning up test environment..."
dropdb test_recovery_db
rm -rf $TEST_ENV

echo "Recovery test complete"
```

---

## 🔐 **Security Procedures**

### Daily Security Check
```bash
#!/bin/bash
# daily_security_check.sh

echo "=== Daily Security Check ==="

# 1. Check for unauthorized access attempts
echo "1. Checking access logs..."
grep -i "failed\|unauthorized\|denied" /var/log/auth.log | tail -20

# 2. Verify service configurations
echo "2. Verifying security configurations..."
./scripts/security_config_check.sh

# 3. Check file permissions
echo "3. Checking file permissions..."
find /opt/hybrid-pipeline -type f -perm /o+w | head -10

# 4. Monitor network connections
echo "4. Monitoring network connections..."
netstat -tuln | grep :808[0-9]

# 5. Check for security updates
echo "5. Checking for security updates..."
./scripts/security_update_check.sh

echo "Daily security check complete"
```

### Security Incident Response
```bash
#!/bin/bash
# security_incident_response.sh

INCIDENT_TYPE=$1

echo "=== SECURITY INCIDENT: $INCIDENT_TYPE ==="

# 1. Immediate containment
case $INCIDENT_TYPE in
    "UNAUTHORIZED_ACCESS")
        # Block suspicious IPs
        ./scripts/block_suspicious_ips.sh
        ;;
    "DATA_BREACH")
        # Isolate affected systems
        ./scripts/isolate_systems.sh
        ;;
    "MALWARE_DETECTED")
        # Quarantine affected components
        ./scripts/quarantine_malware.sh
        ;;
esac

# 2. Preserve evidence
echo "Preserving evidence..."
./scripts/preserve_evidence.sh --incident-type=$INCIDENT_TYPE

# 3. Assess impact
echo "Assessing impact..."
./scripts/security_impact_assessment.sh

# 4. Notify stakeholders
echo "Notifying stakeholders..."
./scripts/security_notification.sh --incident-type=$INCIDENT_TYPE

# 5. Begin recovery
echo "Initiating security recovery..."
./scripts/security_recovery.sh --incident-type=$INCIDENT_TYPE

echo "Security incident response initiated"
```

---

## 📋 **Change Management Procedures**

### Pre-Deployment Validation
```bash
#!/bin/bash
# pre_deployment_validation.sh

DEPLOYMENT_PACKAGE=$1

echo "=== Pre-Deployment Validation ==="

# 1. Package integrity check
echo "1. Checking package integrity..."
./scripts/package_integrity_check.sh $DEPLOYMENT_PACKAGE

# 2. Compatibility testing
echo "2. Testing compatibility..."
./scripts/compatibility_test.sh $DEPLOYMENT_PACKAGE

# 3. Security scan
echo "3. Running security scan..."
./scripts/security_scan.sh $DEPLOYMENT_PACKAGE

# 4. Performance impact assessment
echo "4. Assessing performance impact..."
./scripts/performance_impact_assessment.sh $DEPLOYMENT_PACKAGE

# 5. Rollback plan validation
echo "5. Validating rollback plan..."
./scripts/rollback_plan_validation.sh

echo "Pre-deployment validation complete"
```

### Blue-Green Deployment
```bash
#!/bin/bash
# blue_green_deployment.sh

DEPLOYMENT_VERSION=$1
CURRENT_ENV="blue"  # or "green"
TARGET_ENV="green"  # or "blue"

echo "=== Blue-Green Deployment: $DEPLOYMENT_VERSION ==="

# 1. Prepare target environment
echo "1. Preparing $TARGET_ENV environment..."
./scripts/prepare_environment.sh --env=$TARGET_ENV --version=$DEPLOYMENT_VERSION

# 2. Deploy to target environment
echo "2. Deploying to $TARGET_ENV..."
./scripts/deploy_to_environment.sh --env=$TARGET_ENV --version=$DEPLOYMENT_VERSION

# 3. Validate deployment
echo "3. Validating deployment..."
./scripts/validate_deployment.sh --env=$TARGET_ENV

# 4. Run smoke tests
echo "4. Running smoke tests..."
./scripts/smoke_tests.sh --env=$TARGET_ENV

# 5. Gradual traffic switch (10% -> 50% -> 100%)
echo "5. Switching traffic..."
./scripts/traffic_switch.sh --from=$CURRENT_ENV --to=$TARGET_ENV --percentage=10
sleep 300  # Monitor for 5 minutes
./scripts/traffic_switch.sh --from=$CURRENT_ENV --to=$TARGET_ENV --percentage=50
sleep 300
./scripts/traffic_switch.sh --from=$CURRENT_ENV --to=$TARGET_ENV --percentage=100

# 6. Post-deployment validation
echo "6. Post-deployment validation..."
./scripts/post_deployment_validation.sh

echo "Blue-green deployment complete"
```

---

## 📞 **Escalation Procedures**

### Escalation Matrix
```
Level 1: Automated Response
├── High memory usage (80-90%)
├── Elevated temperature (75-85°C)
├── Moderate throughput decline (25-30 docs/hour)
└── Action: Automated optimization

Level 2: Operations Team
├── Very high memory usage (90-95%)
├── High temperature (85-90°C)
├── Significant throughput decline (<25 docs/hour)
├── Quality degradation (<0.75)
└── Action: Manual intervention within 15 minutes

Level 3: Engineering Team
├── Critical memory usage (>95%)
├── Critical temperature (>90°C)
├── Service failures
├── Major quality issues (<0.70)
└── Action: Emergency response within 5 minutes

Level 4: Management
├── Extended outages (>1 hour)
├── Data integrity issues
├── Security incidents
└── Action: Immediate escalation
```

### On-Call Procedures
```bash
#!/bin/bash
# on_call_procedures.sh

ALERT_LEVEL=$1
COMPONENT=$2

case $ALERT_LEVEL in
    "P1")  # Critical
        # Page immediately
        ./scripts/page_oncall.sh --severity=critical --component=$COMPONENT
        # Execute immediate response
        ./scripts/critical_response.sh --component=$COMPONENT
        ;;
    "P2")  # High
        # Send alert to on-call
        ./scripts/alert_oncall.sh --severity=high --component=$COMPONENT
        # Execute standard response
        ./scripts/high_priority_response.sh --component=$COMPONENT
        ;;
    "P3")  # Medium
        # Log to ticket system
        ./scripts/create_ticket.sh --severity=medium --component=$COMPONENT
        ;;
esac
```

---

## 📊 **Reporting and Documentation**

### Daily Operations Report
```bash
#!/bin/bash
# daily_operations_report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/opt/hybrid-pipeline/reports/daily_$REPORT_DATE.json"

echo "Generating daily operations report for $REPORT_DATE..."

python scripts/generate_daily_report.py \
    --date=$REPORT_DATE \
    --output=$REPORT_FILE \
    --include-performance \
    --include-quality \
    --include-issues \
    --include-recommendations

echo "Daily report generated: $REPORT_FILE"
```

### Weekly Performance Summary
```python
#!/usr/bin/env python3
# weekly_performance_summary.py

import json
from datetime import datetime, timedelta
from pathlib import Path

def generate_weekly_summary():
    """Generate comprehensive weekly performance summary"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    summary = {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "performance": {
            "avg_throughput": get_avg_throughput(start_date, end_date),
            "avg_quality": get_avg_quality(start_date, end_date),
            "uptime_percentage": get_uptime_percentage(start_date, end_date),
            "error_rate": get_error_rate(start_date, end_date)
        },
        "resource_usage": {
            "avg_cpu_usage": get_avg_cpu_usage(start_date, end_date),
            "avg_memory_usage": get_avg_memory_usage(start_date, end_date),
            "peak_temperature": get_peak_temperature(start_date, end_date)
        },
        "incidents": get_incidents(start_date, end_date),
        "recommendations": get_weekly_recommendations()
    }
    
    # Save report
    report_file = f"/opt/hybrid-pipeline/reports/weekly_{start_date.strftime('%Y-%m-%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Weekly summary generated: {report_file}")
    return summary

if __name__ == "__main__":
    generate_weekly_summary()
```

---

**✅ Operational Procedures Complete**  
**🎯 Enterprise-Ready Production Operations**  
**📊 Comprehensive Monitoring and Maintenance Framework**  
**🚀 Weeks 5-8 Advanced Implementation Operational Excellence**