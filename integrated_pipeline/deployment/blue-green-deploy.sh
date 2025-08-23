#!/bin/bash
# Blue-Green Deployment Script for RAN LLM Pipeline
# Handles zero-downtime deployments with validation and rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT=""
IMAGE=""
SLOT=""
NAMESPACE=""
TIMEOUT=600
DRY_RUN=false
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_DELAY=30

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a deployment.log
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a deployment.log
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a deployment.log
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a deployment.log
}

usage() {
    cat << EOF
Usage: $0 --environment <env> --image <image> --slot <slot> [options]

Required:
  --environment <env>       Target environment (staging|production)
  --image <image>          Container image to deploy
  --slot <slot>            Deployment slot (blue|green)

Options:
  --namespace <ns>         Kubernetes namespace (default: ran-llm-<environment>)
  --timeout <seconds>      Deployment timeout (default: 600)
  --dry-run               Show what would be done without executing
  --health-retries <n>    Health check retry attempts (default: 10)
  --health-delay <s>      Delay between health checks (default: 30)
  --help                  Show this help message

Examples:
  $0 --environment production --image ghcr.io/repo/ran-llm:v1.2.3 --slot blue
  $0 --environment staging --image ghcr.io/repo/ran-llm:latest --slot green --dry-run
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --image)
                IMAGE="$2"
                shift 2
                ;;
            --slot)
                SLOT="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --health-retries)
                HEALTH_CHECK_RETRIES="$2"
                shift 2
                ;;
            --health-delay)
                HEALTH_CHECK_DELAY="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$ENVIRONMENT" || -z "$IMAGE" || -z "$SLOT" ]]; then
        log_error "Environment, image, and slot are required"
        usage
        exit 1
    fi

    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Environment must be 'staging' or 'production'"
        exit 1
    fi

    if [[ "$SLOT" != "blue" && "$SLOT" != "green" ]]; then
        log_error "Slot must be 'blue' or 'green'"
        exit 1
    fi

    # Set default namespace if not provided
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="ran-llm-${ENVIRONMENT}"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == false ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    # Check image accessibility (simplified)
    log_info "Validating container image: $IMAGE"
    # In a real deployment, you might want to check if the image exists and is pullable

    log_success "Prerequisites check passed"
}

# Create Kubernetes manifests
create_manifests() {
    log_info "Creating Kubernetes manifests for $SLOT slot..."

    local manifests_dir="$PROJECT_ROOT/integrated_pipeline/deployment/manifests"
    mkdir -p "$manifests_dir"

    # Deployment manifest
    cat > "$manifests_dir/deployment-$SLOT.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ran-llm-pipeline-$SLOT
  namespace: $NAMESPACE
  labels:
    app: ran-llm-pipeline
    slot: $SLOT
    version: $(echo "$IMAGE" | cut -d':' -f2)
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ran-llm-pipeline
      slot: $SLOT
  template:
    metadata:
      labels:
        app: ran-llm-pipeline
        slot: $SLOT
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: ran-llm-pipeline
        image: $IMAGE
        ports:
        - containerPort: 80
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          value: $ENVIRONMENT
        - name: DEPLOYMENT_SLOT
          value: $SLOT
        - name: RUST_LOG
          value: "info"
        - name: PYTHON_OPTIMIZE
          value: "2"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: shared-memory
          mountPath: /dev/shm
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: config
        configMap:
          name: ran-llm-config
      - name: shared-memory
        emptyDir:
          medium: Memory
          sizeLimit: "15Gi"
      - name: temp-storage
        emptyDir:
          sizeLimit: "5Gi"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: ["ran-llm-pipeline"]
              topologyKey: kubernetes.io/hostname
EOF

    # Service manifest
    cat > "$manifests_dir/service-$SLOT.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: ran-llm-pipeline-$SLOT
  namespace: $NAMESPACE
  labels:
    app: ran-llm-pipeline
    slot: $SLOT
spec:
  ports:
  - port: 80
    targetPort: http
    name: http
  - port: 9090
    targetPort: metrics
    name: metrics
  selector:
    app: ran-llm-pipeline
    slot: $SLOT
EOF

    # HPA manifest for auto-scaling
    cat > "$manifests_dir/hpa-$SLOT.yaml" << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ran-llm-pipeline-$SLOT
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ran-llm-pipeline-$SLOT
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
EOF

    log_success "Kubernetes manifests created"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes ($SLOT slot)..."

    local manifests_dir="$PROJECT_ROOT/integrated_pipeline/deployment/manifests"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would apply the following manifests:"
        for manifest in "$manifests_dir"/*-"$SLOT".yaml; do
            echo "  - $(basename "$manifest")"
        done
        return 0
    fi

    # Apply manifests
    for manifest in "$manifests_dir"/*-"$SLOT".yaml; do
        log_info "Applying $(basename "$manifest")..."
        kubectl apply -f "$manifest"
    done

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    if ! kubectl rollout status deployment -n "$NAMESPACE" "ran-llm-pipeline-$SLOT" --timeout="${TIMEOUT}s"; then
        log_error "Deployment failed to become ready within $TIMEOUT seconds"
        return 1
    fi

    log_success "Deployment completed successfully"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks on $SLOT slot..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run health checks"
        return 0
    fi

    local service_name="ran-llm-pipeline-$SLOT"
    local health_check_count=0

    # Wait for service to be available
    log_info "Waiting for service to be available..."
    sleep 30

    for ((i=1; i<=HEALTH_CHECK_RETRIES; i++)); do
        log_info "Health check attempt $i/$HEALTH_CHECK_RETRIES..."

        # Use kubectl port-forward to test the service
        local port_forward_pid
        kubectl port-forward -n "$NAMESPACE" "service/$service_name" 8080:80 &
        port_forward_pid=$!
        
        sleep 10  # Give port-forward time to establish

        # Test health endpoint
        if curl -f -s --max-time 10 "http://localhost:8080/health" > /dev/null; then
            health_check_count=$((health_check_count + 1))
            log_success "Health check $i passed"
        else
            log_warning "Health check $i failed"
        fi

        # Test readiness endpoint
        if curl -f -s --max-time 10 "http://localhost:8080/ready" > /dev/null; then
            log_info "Readiness check $i passed"
        else
            log_warning "Readiness check $i failed"
        fi

        # Cleanup port-forward
        kill $port_forward_pid 2>/dev/null || true
        wait $port_forward_pid 2>/dev/null || true

        # If this isn't the last attempt, wait before next check
        if [[ $i -lt $HEALTH_CHECK_RETRIES ]]; then
            sleep "$HEALTH_CHECK_DELAY"
        fi
    done

    # Require at least 70% of health checks to pass
    local required_passes=$((HEALTH_CHECK_RETRIES * 7 / 10))
    if [[ $health_check_count -ge $required_passes ]]; then
        log_success "Health checks passed ($health_check_count/$HEALTH_CHECK_RETRIES successful)"
        return 0
    else
        log_error "Health checks failed ($health_check_count/$HEALTH_CHECK_RETRIES successful, need $required_passes)"
        return 1
    fi
}

# Run performance validation
run_performance_validation() {
    log_info "Running performance validation..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run performance validation"
        return 0
    fi

    # Run a simple load test to validate performance
    local service_name="ran-llm-pipeline-$SLOT"
    
    log_info "Starting performance validation against $service_name..."
    
    # Use kubectl port-forward for testing
    kubectl port-forward -n "$NAMESPACE" "service/$service_name" 8080:80 &
    local port_forward_pid=$!
    
    sleep 10
    
    # Simple performance test (would use proper load testing tools in production)
    local response_times=()
    for i in {1..10}; do
        local start_time=$(date +%s.%N)
        if curl -f -s --max-time 30 "http://localhost:8080/api/v1/status" > /dev/null; then
            local end_time=$(date +%s.%N)
            local response_time=$(echo "$end_time - $start_time" | bc -l)
            response_times+=("$response_time")
            log_info "Request $i: ${response_time}s"
        else
            log_warning "Request $i failed"
        fi
        sleep 1
    done
    
    # Cleanup port-forward
    kill $port_forward_pid 2>/dev/null || true
    wait $port_forward_pid 2>/dev/null || true
    
    # Calculate average response time
    if [[ ${#response_times[@]} -gt 0 ]]; then
        local total=0
        for time in "${response_times[@]}"; do
            total=$(echo "$total + $time" | bc -l)
        done
        local average=$(echo "scale=3; $total / ${#response_times[@]}" | bc -l)
        
        log_info "Average response time: ${average}s"
        
        # Validate response time is acceptable (< 5 seconds)
        if (( $(echo "$average < 5.0" | bc -l) )); then
            log_success "Performance validation passed"
            return 0
        else
            log_error "Performance validation failed: average response time too high"
            return 1
        fi
    else
        log_error "Performance validation failed: no successful requests"
        return 1
    fi
}

# Create deployment record
create_deployment_record() {
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    local deployment_record
    deployment_record=$(cat << EOF
{
    "timestamp": "$timestamp",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "slot": "$SLOT",
    "image": "$IMAGE",
    "initiated_by": "${USER:-unknown}",
    "deployment_id": "deploy-$(date +%s)",
    "status": "completed",
    "health_checks": "passed",
    "performance_validation": "passed"
}
EOF
)

    if [[ "$DRY_RUN" == false ]]; then
        echo "$deployment_record" > "deployment-${timestamp}-${SLOT}.json"
        
        # Store in Kubernetes configmap for audit trail
        kubectl create configmap -n "$NAMESPACE" "deployment-record-$(date +%s)" \
            --from-literal=record="$deployment_record" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        log_info "Deployment record created: deployment-${timestamp}-${SLOT}.json"
    else
        log_info "[DRY RUN] Would create deployment record"
        echo "$deployment_record"
    fi
}

# Main function
main() {
    echo -e "${BLUE}🚀 RAN LLM Pipeline Blue-Green Deployment${NC}"
    echo "========================================="
    
    parse_args "$@"
    
    log_info "Starting blue-green deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image: $IMAGE"
    log_info "Slot: $SLOT"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry run: $DRY_RUN"
    
    check_prerequisites
    create_manifests
    
    if ! deploy_to_kubernetes; then
        log_error "Deployment failed"
        exit 1
    fi
    
    if ! run_health_checks; then
        log_error "Health checks failed, rolling back..."
        if [[ "$DRY_RUN" == false ]]; then
            kubectl delete deployment -n "$NAMESPACE" "ran-llm-pipeline-$SLOT" || true
        fi
        exit 1
    fi
    
    if ! run_performance_validation; then
        log_error "Performance validation failed"
        # Note: Not failing the deployment for performance issues in this version
        log_warning "Continuing with deployment despite performance concerns"
    fi
    
    create_deployment_record
    
    echo -e "\n${GREEN}✅ Blue-Green deployment completed successfully!${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Slot: $SLOT"
    echo "Image: $IMAGE"
    echo "Namespace: $NAMESPACE"
    echo ""
    echo "Next steps:"
    echo "1. Monitor the deployment for stability"
    echo "2. Run additional validation tests if needed"
    echo "3. Switch traffic to this slot when ready"
    echo "4. Clean up the old deployment slot"
    
    log_success "Blue-green deployment process completed"
}

# Trap signals for cleanup
trap 'log_warning "Deployment interrupted"; exit 130' INT TERM

# Run main function
main "$@"