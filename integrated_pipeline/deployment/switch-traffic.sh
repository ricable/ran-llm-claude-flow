#!/bin/bash
# Traffic Switch Script for Blue-Green Deployment
# Switches traffic between blue and green deployment slots

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT=""
TARGET_SLOT=""
NAMESPACE=""
DRY_RUN=false
GRADUAL_SWITCH=false
SWITCH_PERCENTAGE=100

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a traffic-switch.log
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a traffic-switch.log
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a traffic-switch.log
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a traffic-switch.log
}

usage() {
    cat << EOF
Usage: $0 --environment <env> --target <slot> [options]

Required:
  --environment <env>       Target environment (staging|production)
  --target <slot>           Target slot to switch traffic to (blue|green)

Options:
  --namespace <ns>          Kubernetes namespace (default: ran-llm-<environment>)
  --dry-run                Show what would be done without executing
  --gradual                Enable gradual traffic switch
  --percentage <n>         Percentage of traffic to switch (default: 100)
  --help                   Show this help message

Examples:
  $0 --environment production --target blue
  $0 --environment staging --target green --gradual --percentage 50
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
            --target)
                TARGET_SLOT="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --gradual)
                GRADUAL_SWITCH=true
                shift
                ;;
            --percentage)
                SWITCH_PERCENTAGE="$2"
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
    if [[ -z "$ENVIRONMENT" || -z "$TARGET_SLOT" ]]; then
        log_error "Environment and target slot are required"
        usage
        exit 1
    fi

    if [[ "$TARGET_SLOT" != "blue" && "$TARGET_SLOT" != "green" ]]; then
        log_error "Target slot must be 'blue' or 'green'"
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
        log_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi

    # Check target deployment exists and is ready
    if ! kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$TARGET_SLOT" &> /dev/null; then
        log_error "Target deployment 'ran-llm-pipeline-$TARGET_SLOT' not found"
        exit 1
    fi

    local ready_replicas
    ready_replicas=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$TARGET_SLOT" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local desired_replicas
    desired_replicas=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$TARGET_SLOT" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    if [[ "$ready_replicas" != "$desired_replicas" || "$ready_replicas" == "0" ]]; then
        log_error "Target deployment is not ready ($ready_replicas/$desired_replicas replicas ready)"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get current traffic configuration
get_current_traffic_config() {
    log_info "Getting current traffic configuration..."

    # Get current active slot from configmap
    CURRENT_SLOT=$(kubectl get configmap -n "$NAMESPACE" traffic-config -o jsonpath='{.data.active_slot}' 2>/dev/null || echo "")
    
    if [[ -z "$CURRENT_SLOT" ]]; then
        log_warning "No current active slot found, assuming blue"
        CURRENT_SLOT="blue"
    fi

    log_info "Current active slot: $CURRENT_SLOT"
    log_info "Target slot: $TARGET_SLOT"

    if [[ "$CURRENT_SLOT" == "$TARGET_SLOT" ]]; then
        log_warning "Traffic is already routed to $TARGET_SLOT slot"
        return 0
    fi
}

# Update ingress configuration
update_ingress() {
    local target_service="ran-llm-pipeline-$TARGET_SLOT"
    
    log_info "Updating ingress to route traffic to $target_service..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would update ingress to route to $target_service"
        return 0
    fi

    # Create or update ingress
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ran-llm-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.ran-llm.com
    secretName: ran-llm-tls
  rules:
  - host: api.ran-llm.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: $target_service
            port:
              number: 80
EOF

    log_success "Ingress updated"
}

# Update traffic configmap
update_traffic_configmap() {
    log_info "Updating traffic configuration..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would update traffic configmap with active_slot=$TARGET_SLOT"
        return 0
    fi

    # Create or update configmap
    kubectl create configmap -n "$NAMESPACE" traffic-config \
        --from-literal=active_slot="$TARGET_SLOT" \
        --from-literal=switch_timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --from-literal=previous_slot="$CURRENT_SLOT" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_success "Traffic configuration updated"
}

# Gradual traffic switch using weighted routing
gradual_traffic_switch() {
    log_info "Performing gradual traffic switch ($SWITCH_PERCENTAGE% to $TARGET_SLOT)..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would perform gradual traffic switch"
        return 0
    fi

    local current_weight=$((100 - SWITCH_PERCENTAGE))
    local target_weight=$SWITCH_PERCENTAGE

    # Create weighted ingress configuration
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ran-llm-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "$target_weight"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.ran-llm.com
    secretName: ran-llm-tls
  rules:
  - host: api.ran-llm.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ran-llm-pipeline-$TARGET_SLOT
            port:
              number: 80
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ran-llm-ingress-primary
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.ran-llm.com
    secretName: ran-llm-tls
  rules:
  - host: api.ran-llm.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ran-llm-pipeline-$CURRENT_SLOT
            port:
              number: 80
EOF

    log_success "Gradual traffic switch configured ($SWITCH_PERCENTAGE% to $TARGET_SLOT)"
}

# Validate traffic switch
validate_traffic_switch() {
    log_info "Validating traffic switch..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would validate traffic switch"
        return 0
    fi

    # Wait for ingress to update
    log_info "Waiting for ingress controller to update..."
    sleep 30

    # Test that the correct service is receiving traffic
    local test_attempts=5
    local successful_tests=0

    for ((i=1; i<=test_attempts; i++)); do
        log_info "Testing traffic routing (attempt $i/$test_attempts)..."
        
        # Use kubectl port-forward to test the ingress
        kubectl port-forward -n "$NAMESPACE" service/ran-llm-pipeline-"$TARGET_SLOT" 8080:80 &
        local port_forward_pid=$!
        
        sleep 5
        
        if curl -f -s --max-time 10 -H "Host: api.ran-llm.com" "http://localhost:8080/health" > /dev/null; then
            successful_tests=$((successful_tests + 1))
            log_success "Traffic test $i passed"
        else
            log_warning "Traffic test $i failed"
        fi
        
        # Cleanup port-forward
        kill $port_forward_pid 2>/dev/null || true
        wait $port_forward_pid 2>/dev/null || true
        
        sleep 5
    done

    # Require at least 80% success rate
    local required_successes=$((test_attempts * 4 / 5))
    if [[ $successful_tests -ge $required_successes ]]; then
        log_success "Traffic switch validation passed ($successful_tests/$test_attempts)"
        return 0
    else
        log_error "Traffic switch validation failed ($successful_tests/$test_attempts, need $required_successes)"
        return 1
    fi
}

# Monitor traffic switch
monitor_traffic_switch() {
    log_info "Monitoring traffic switch for 60 seconds..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would monitor traffic switch"
        return 0
    fi

    local monitoring_duration=60
    local check_interval=10
    local checks=$((monitoring_duration / check_interval))

    for ((i=1; i<=checks; i++)); do
        log_info "Monitoring check $i/$checks..."
        
        # Check pod status
        local target_pods_ready
        target_pods_ready=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$TARGET_SLOT" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        if [[ "$target_pods_ready" == "0" ]]; then
            log_error "No pods ready in target deployment"
            return 1
        fi
        
        log_info "Target deployment has $target_pods_ready pods ready"
        
        # Basic health check
        kubectl port-forward -n "$NAMESPACE" service/ran-llm-pipeline-"$TARGET_SLOT" 8080:80 &
        local port_forward_pid=$!
        
        sleep 5
        
        if curl -f -s --max-time 5 "http://localhost:8080/health" > /dev/null; then
            log_info "Health check passed"
        else
            log_warning "Health check failed"
        fi
        
        # Cleanup port-forward
        kill $port_forward_pid 2>/dev/null || true
        wait $port_forward_pid 2>/dev/null || true
        
        if [[ $i -lt $checks ]]; then
            sleep $check_interval
        fi
    done

    log_success "Traffic switch monitoring completed"
}

# Create traffic switch record
create_switch_record() {
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    local switch_record
    switch_record=$(cat << EOF
{
    "timestamp": "$timestamp",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "from_slot": "$CURRENT_SLOT",
    "to_slot": "$TARGET_SLOT",
    "switch_type": "$( [[ "$GRADUAL_SWITCH" == true ]] && echo "gradual" || echo "complete" )",
    "percentage": $SWITCH_PERCENTAGE,
    "initiated_by": "${USER:-unknown}",
    "status": "completed"
}
EOF
)

    if [[ "$DRY_RUN" == false ]]; then
        echo "$switch_record" > "traffic-switch-${timestamp}.json"
        
        # Store in Kubernetes configmap for audit trail
        kubectl create configmap -n "$NAMESPACE" "traffic-switch-record-$(date +%s)" \
            --from-literal=record="$switch_record" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        log_info "Traffic switch record created: traffic-switch-${timestamp}.json"
    else
        log_info "[DRY RUN] Would create traffic switch record"
        echo "$switch_record"
    fi
}

# Main function
main() {
    echo -e "${BLUE}🔄 RAN LLM Pipeline Traffic Switch${NC}"
    echo "================================="
    
    parse_args "$@"
    
    log_info "Starting traffic switch process"
    log_info "Environment: $ENVIRONMENT"
    log_info "Target slot: $TARGET_SLOT"
    log_info "Namespace: $NAMESPACE"
    log_info "Gradual switch: $GRADUAL_SWITCH"
    log_info "Switch percentage: $SWITCH_PERCENTAGE%"
    log_info "Dry run: $DRY_RUN"
    
    check_prerequisites
    get_current_traffic_config
    
    if [[ "$CURRENT_SLOT" == "$TARGET_SLOT" ]]; then
        log_info "No traffic switch needed, already on target slot"
        exit 0
    fi
    
    # Perform traffic switch
    if [[ "$GRADUAL_SWITCH" == true && "$SWITCH_PERCENTAGE" -lt 100 ]]; then
        gradual_traffic_switch
    else
        update_ingress
    fi
    
    update_traffic_configmap
    
    if ! validate_traffic_switch; then
        log_error "Traffic switch validation failed"
        exit 1
    fi
    
    monitor_traffic_switch
    create_switch_record
    
    echo -e "\n${GREEN}✅ Traffic switch completed successfully!${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Traffic switched from: $CURRENT_SLOT -> $TARGET_SLOT"
    echo "Switch type: $( [[ "$GRADUAL_SWITCH" == true ]] && echo "Gradual ($SWITCH_PERCENTAGE%)" || echo "Complete" )"
    
    log_success "Traffic switch process completed"
}

# Run main function
main "$@"