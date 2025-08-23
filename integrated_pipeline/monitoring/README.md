# High-Performance Pipeline Monitoring System

A comprehensive real-time monitoring system designed for the hybrid Rust-Python document processing pipeline, providing <1% overhead metrics collection, 5-second bottleneck detection, and automatic performance optimization.

## 🚀 Key Features

### Real-time Metrics Collection (Sub-1% Overhead)
- **High-frequency monitoring**: 100ms collection intervals
- **Lock-free counters**: Atomic operations for zero-contention updates
- **20+ performance metrics**: CPU, memory, IPC, throughput, errors
- **Memory-efficient storage**: Ring buffers with configurable history

### Intelligent Bottleneck Detection (5-second detection)
- **Pattern recognition**: ML-based anomaly detection
- **Multi-dimensional analysis**: CPU, memory, I/O, IPC bottlenecks
- **Predictive alerting**: Detect issues before they impact performance
- **Impact scoring**: Quantified bottleneck severity (0-1 scale)

### Adaptive Performance Optimization
- **Real-time adjustments**: Dynamic concurrency and resource allocation
- **Memory rebalancing**: Automatic pool size optimization
- **IPC optimization**: Message batching and compression
- **Model selection**: Automatic switching to optimal inference models

### Real-time Dashboard & Visualization
- **WebSocket updates**: Sub-second refresh rates
- **Interactive charts**: Throughput, resource utilization trends
- **Alert management**: Real-time bottleneck and optimization status
- **Historical analysis**: Multiple time ranges (5min to 4hr)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring System                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  MetricsCollector │ BottleneckAnalyzer │    AdaptiveOptimizer    │
│  - Lock-free     │ - Pattern detect   │ - Real-time tuning      │
│  - <1% overhead  │ - 5s detection     │ - Auto-scaling          │
│  - 20+ metrics   │ - ML-based         │ - Memory rebalancing    │
└─────────────────┴─────────────────┴─────────────────────────────┘
            │                  │                       │
            ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Dashboard Server                            │
│  - Real-time WebSocket updates                                  │
│  - Interactive performance charts                               │
│  - Alert management interface                                   │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Alert System                               │
│  - Multi-channel alerting (log, webhook, email)                │
│  - Intelligent suppression (cooldown, rate limiting)           │
│  - Severity-based routing                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|--------------------|
| Document Processing Rate | 20-30 docs/hour | < 10 docs/hour |
| Rust Memory Usage | < 60GB | > 70GB |
| Python Memory Usage | < 45GB | > 55GB |
| Shared Memory Usage | < 15GB | > 20GB |
| IPC Latency P99 | < 10ms | > 50ms |
| CPU Utilization | < 85% | > 95% |
| Error Rate | < 1% | > 5% |
| System Health Score | > 80% | < 60% |

## 🚀 Quick Start

### 1. Build the Monitoring System
```bash
cd integrated_pipeline/monitoring
cargo build --release
```

### 2. Run Standalone Metrics Collector
```bash
# Basic monitoring with default config
cargo run --bin monitoring-collector

# Custom configuration
cargo run --bin monitoring-collector -- --config config/monitoring.yaml

# With Prometheus endpoint
cargo run --bin monitoring-collector -- --prometheus --prometheus-port 9090

# Verbose logging for debugging
cargo run --bin monitoring-collector -- --verbose
```

### 3. Launch Dashboard Server
```bash
# Start dashboard with default settings
cargo run --bin monitoring-dashboard

# Custom port and configuration
cargo run --bin monitoring-dashboard -- --config config/monitoring.yaml --port 8080

# Demo mode with simulated data
cargo run --bin monitoring-dashboard -- --demo

# Production mode
cargo run --release --bin monitoring-dashboard -- --config production.yaml
```

### 4. Access Real-time Dashboard
Open your browser to `http://localhost:8080` for the full monitoring interface.

## 📈 Monitoring Integration

### Rust Integration
```rust
use monitoring::{MonitoringSystem, config::MonitoringConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize monitoring
    let config = MonitoringConfig::load()?;
    let mut monitoring = MonitoringSystem::with_config(config).await?;
    monitoring.start().await?;

    // Your application logic here
    loop {
        // Processing logic...
        monitoring.metrics_collector.increment_documents_processed();
        monitoring.metrics_collector.record_ipc_latency(latency_ms);
    }
    
    monitoring.shutdown().await?;
    Ok(())
}
```

### Python Integration via FFI
```python
import ctypes

# Load monitoring library
monitoring_lib = ctypes.CDLL('./target/release/libmonitoring.so')

# Increment counters
monitoring_lib.increment_documents_processed()
monitoring_lib.record_processing_time(c_double(processing_time_ms))
```

## 🔧 Configuration

### Environment Variables
```bash
export MONITORING_CONFIG_PATH="./config/monitoring.yaml"
export RUST_LOG="monitoring=info"
export MONITORING_DASHBOARD_PORT=8080
```

### Configuration File Structure
```yaml
# Core collection settings
collection_interval_ms: 100
analysis_interval_ms: 1000
max_history_size: 10000

# Dashboard settings
dashboard:
  enabled: true
  port: 8080
  websocket_enabled: true

# Alert thresholds
alerts:
  severity_thresholds:
    cpu_utilization: 85.0
    memory_utilization: 90.0
    ipc_latency_ms: 10.0

# Performance targets
targets:
  document_processing_rate: 25.0
  rust_memory_limit: 60.0
  python_memory_limit: 45.0
```

## 🧪 Performance Benchmarks

Run comprehensive benchmarks to verify monitoring overhead:

```bash
# Metrics collection performance
cargo bench --bench metrics_collection

# Bottleneck detection accuracy and speed
cargo bench --bench bottleneck_detection

# Full system benchmarks
cargo bench
```

### Benchmark Results (Target)
- **Metrics Collection**: < 0.5% CPU overhead
- **Memory Usage**: < 100MB for 10,000 metric snapshots  
- **Bottleneck Detection**: < 5 seconds for 1,000 metrics
- **Dashboard Response**: < 100ms API response time
- **WebSocket Updates**: < 50ms delivery latency

## 🚨 Alerting

### Alert Channels
1. **Log Alerts**: Structured log entries for log aggregation systems
2. **Webhook Alerts**: HTTP POST to external monitoring systems
3. **Email Alerts**: SMTP-based notifications for critical issues

### Alert Types
- **Threshold Alerts**: Metric exceeds configured limits
- **Bottleneck Alerts**: Performance bottlenecks detected  
- **System Alerts**: Infrastructure-level issues
- **Trend Alerts**: Performance regression detection

### Alert Suppression
- **Cooldown periods**: Prevent alert spam (default: 5 minutes)
- **Rate limiting**: Maximum alerts per minute (default: 10)
- **Severity filtering**: Channel-specific minimum severity levels

## 🔍 API Endpoints

### REST API
- `GET /api/metrics` - Current system metrics
- `GET /api/bottlenecks` - Active performance bottlenecks
- `GET /api/optimizations` - Current optimization status
- `GET /api/dashboard` - Dashboard state
- `GET /api/health` - Health check endpoint

### WebSocket API
- `/ws` - Real-time metric updates
- Message types: `DashboardUpdate`, `Alert`, `OptimizationApplied`

### Prometheus Integration
- `/metrics` - Prometheus-formatted metrics
- Compatible with Grafana dashboards
- Custom metric labels for pipeline stages

## 🛠️ Development

### Running Tests
```bash
cargo test --all-features
cargo test --test integration_tests
```

### Code Coverage
```bash
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

### Profiling
```bash
cargo install flamegraph
cargo flamegraph --bin monitoring-collector
```

## 📚 Advanced Usage

### Custom Bottleneck Detection
```rust
use monitoring::bottleneck_analyzer::{BottleneckAnalyzer, BottleneckPattern};

let mut analyzer = BottleneckAnalyzer::new(&config)?;

// Add custom bottleneck pattern
let custom_pattern = BottleneckPattern::new()
    .detection_threshold(90.0)
    .confidence_threshold(0.8)
    .duration_threshold(Duration::minutes(2));

analyzer.add_custom_pattern("custom_bottleneck", custom_pattern);
```

### Custom Optimization Strategies
```rust
use monitoring::optimizer::{AdaptiveOptimizer, OptimizationStrategy};

let mut optimizer = AdaptiveOptimizer::new(&config)?;

// Implement custom optimization strategy
struct CustomStrategy;
impl OptimizationStrategy for CustomStrategy {
    async fn analyze(&self, metrics: &SystemMetrics) -> Result<Vec<Optimization>> {
        // Custom optimization logic
        Ok(vec![])
    }
}

optimizer.add_strategy(Box::new(CustomStrategy));
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin monitoring-dashboard

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/monitoring-dashboard /usr/local/bin/
COPY --from=builder /app/config/ /etc/monitoring/
EXPOSE 8080
CMD ["monitoring-dashboard", "--config", "/etc/monitoring/production.yaml"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pipeline-monitoring
  template:
    metadata:
      labels:
        app: pipeline-monitoring
    spec:
      containers:
      - name: monitoring
        image: pipeline-monitoring:latest
        ports:
        - containerPort: 8080
        env:
        - name: MONITORING_CONFIG_PATH
          value: "/etc/monitoring/production.yaml"
        volumeMounts:
        - name: config
          mountPath: /etc/monitoring
      volumes:
      - name: config
        configMap:
          name: monitoring-config
```

### Production Checklist
- [ ] Configure appropriate alert thresholds for your environment
- [ ] Set up log rotation and retention policies  
- [ ] Configure webhook endpoints for external monitoring systems
- [ ] Test alert delivery channels (email, Slack, PagerDuty)
- [ ] Establish monitoring dashboard access controls
- [ ] Set up backup and disaster recovery for monitoring data
- [ ] Configure resource limits and monitoring system scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`cargo test --all-features`)
4. Run benchmarks (`cargo bench`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This monitoring system is part of the integrated pipeline project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing documentation and examples
- Review benchmark results and performance expectations
- Validate configuration against provided schemas

---

**Performance Monitoring System v1.0.0** - Built for the hybrid Rust-Python document processing pipeline with enterprise-grade monitoring capabilities.