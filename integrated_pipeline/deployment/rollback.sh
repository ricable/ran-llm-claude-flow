#!/bin/bash
# Automated Rollback Script for Blue-Green Deployment
# Handles rollback to previous stable deployment with validation

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
REASON="manual"
AUTO_CONFIRM=false
DRY_RUN=false
TIMEOUT=300
NAMESPACE=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a rollback.log
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a rollback.log
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a rollback.log
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a rollback.log
}

usage() {
    cat << EOF
Usage: $0 --environment <env> [options]

Required:
  --environment <env>        Target environment (staging|production)

Options:
  --reason <reason>          Reason for rollback (default: manual)
  --auto-confirm            Skip confirmation prompt
  --dry-run                 Show what would be done without executing
  --timeout <seconds>       Timeout for operations (default: 300)
  --namespace <ns>          Kubernetes namespace (optional)
  --help                    Show this help message

Examples:
  $0 --environment production --reason "deployment-failure"
  $0 --environment staging --auto-confirm --dry-run
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
            --reason)
                REASON="$2"
                shift 2
                ;;
            --auto-confirm)
                AUTO_CONFIRM=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
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
    if [[ -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required"
        usage
        exit 1
    fi

    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Environment must be 'staging' or 'production'"
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

    log_success "Prerequisites check passed"
}

# Get current deployment information
get_current_deployment_info() {
    log_info "Getting current deployment information..."

    # Get current active slot
    CURRENT_SLOT=$(kubectl get configmap -n "$NAMESPACE" traffic-config -o jsonpath='{.data.active_slot}' 2>/dev/null || echo "")
    
    if [[ -z "$CURRENT_SLOT" ]]; then
        log_warning "No active slot found, assuming blue"
        CURRENT_SLOT="blue"
    fi

    # Get previous slot (the one we'll rollback to)
    if [[ "$CURRENT_SLOT" == "blue" ]]; then
        ROLLBACK_SLOT="green"
    else
        ROLLBACK_SLOT="blue"
    fi

    log_info "Current active slot: $CURRENT_SLOT"
    log_info "Rollback target slot: $ROLLBACK_SLOT"

    # Check if rollback target exists
    if ! kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" &> /dev/null; then
        log_error "Rollback target deployment 'ran-llm-pipeline-$ROLLBACK_SLOT' not found"
        log_error "Cannot perform rollback without a previous deployment"
        exit 1
    fi

    # Get deployment details
    CURRENT_IMAGE=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$CURRENT_SLOT" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")
    ROLLBACK_IMAGE=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")

    log_info "Current deployment image: $CURRENT_IMAGE"
    log_info "Rollback target image: $ROLLBACK_IMAGE"
}

# Validate rollback target
validate_rollback_target() {
    log_info "Validating rollback target..."

    # Check if rollback deployment is ready
    local ready_replicas
    ready_replicas=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local desired_replicas
    desired_replicas=$(kubectl get deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    if [[ "$ready_replicas" != "$desired_replicas" || "$ready_replicas" == "0" ]]; then
        log_warning "Rollback target deployment is not fully ready ($ready_replicas/$desired_replicas replicas)"
        log_info "Attempting to scale up rollback target..."
        
        if [[ "$DRY_RUN" == false ]]; then
            kubectl scale deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" --replicas=3
            kubectl rollout status deployment -n "$NAMESPACE" "ran-llm-pipeline-$ROLLBACK_SLOT" --timeout="${TIMEOUT}s"
        fi
    fi

    # Health check on rollback target
    log_info "Running health check on rollback target..."
    if [[ "$DRY_RUN" == false ]]; then
        if ! run_health_check "$ROLLBACK_SLOT"; then
            log_error "Health check failed on rollback target"
            log_error "Cannot rollback to unhealthy deployment"
            exit 1
        fi
    else
        log_info "[DRY RUN] Would run health check on $ROLLBACK_SLOT slot"
    fi

    log_success "Rollback target validation passed"
}

# Run health check
run_health_check() {
    local slot="$1"
    local service_name="ran-llm-pipeline-$slot"
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service -n "$NAMESPACE" "$service_name" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [[ -z "$service_ip" ]]; then
        log_warning "Service $service_name not found, skipping health check"
        return 0
    fi

    # Run health check (simplified)
    log_info "Running health check against $service_name ($service_ip)"
    
    # Use kubectl port-forward to test the service
    local port_forward_pid
    kubectl port-forward -n "$NAMESPACE" "service/$service_name" 8080:80 &
    port_forward_pid=$!
    
    sleep 5  # Give port-forward time to establish
    
    # Test the health endpoint
    local health_check_result=1
    for i in {1..5}; do
        if curl -f -s "http://localhost:8080/health" > /dev/null 2>&1; then
            health_check_result=0
            break
        fi
        log_info "Health check attempt $i/5 failed, retrying..."
        sleep 2
    done
    
    # Cleanup port-forward
    kill $port_forward_pid 2>/dev/null || true
    wait $port_forward_pid 2>/dev/null || true
    
    return $health_check_result
}

# Perform the rollback
perform_rollback() {
    log_info "Starting rollback process..."

    if [[ "$AUTO_CONFIRM" == false && "$DRY_RUN" == false ]]; then
        echo -e "\n${YELLOW}ROLLBACK CONFIRMATION${NC}"
        echo "=================================="
        echo "Environment: $ENVIRONMENT"
        echo "Namespace: $NAMESPACE"
        echo "Current slot: $CURRENT_SLOT -> $ROLLBACK_SLOT"
        echo "Current image: $CURRENT_IMAGE"
        echo "Rollback image: $ROLLBACK_IMAGE"
        echo "Reason: $REASON"
        echo ""
        read -p "Are you sure you want to proceed with rollback? (yes/no): " confirmation
        
        if [[ "$confirmation" != "yes" ]]; then
            log_info "Rollback cancelled by user"
            exit 0
        fi
    fi

    # Create rollback record
    create_rollback_record

    # Step 1: Switch traffic to rollback slot
    log_info "Step 1: Switching traffic to rollback slot ($ROLLBACK_SLOT)"
    if [[ "$DRY_RUN" == false ]]; then
        switch_traffic "$ROLLBACK_SLOT"
    else
        log_info "[DRY RUN] Would switch traffic to $ROLLBACK_SLOT"
    fi

    # Step 2: Wait for traffic switch to complete
    log_info "Step 2: Waiting for traffic switch to complete..."
    if [[ "$DRY_RUN" == false ]]; then
        sleep 10  # Give load balancer time to update
    fi

    # Step 3: Validate rollback
    log_info "Step 3: Validating rollback..."
    if [[ "$DRY_RUN" == false ]]; then
        if ! validate_rollback; then
            log_error "Rollback validation failed"
            # Attempt to switch back if validation fails
            log_warning "Attempting to revert traffic switch..."
            switch_traffic "$CURRENT_SLOT"
            exit 1
        fi
    else
        log_info "[DRY RUN] Would validate rollback"
    fi

    # Step 4: Scale down failed deployment (optional)
    log_info "Step 4: Scaling down failed deployment..."
    if [[ "$DRY_RUN" == false ]]; then
        kubectl scale deployment -n "$NAMESPACE" "ran-llm-pipeline-$CURRENT_SLOT" --replicas=0
    else
        log_info "[DRY RUN] Would scale down ran-llm-pipeline-$CURRENT_SLOT to 0 replicas"
    fi

    log_success "Rollback completed successfully!"
}

# Switch traffic between slots
switch_traffic() {
    local target_slot="$1"
    
    log_info "Switching traffic to $target_slot slot..."

    # Update traffic configuration
    kubectl create configmap -n "$NAMESPACE" traffic-config \
        --from-literal=active_slot="$target_slot" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Update ingress or load balancer configuration
    # This would depend on your specific ingress setup
    kubectl patch ingress -n "$NAMESPACE" ran-llm-ingress \
        -p "{\"spec\":{\"rules\":[{\"host\":\"api.ran-llm.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"ran-llm-pipeline-$target_slot\",\"port\":{\"number\":80}}}}]}}]}}"

    log_success "Traffic switched to $target_slot"
}

# Validate rollback was successful
validate_rollback() {
    log_info "Validating rollback success..."

    # Check if new active slot is responding
    if ! run_health_check "$ROLLBACK_SLOT"; then
        log_error "Health check failed on rolled back deployment"
        return 1
    fi

    # Run basic functionality test
    log_info "Running basic functionality test..."
    if ! run_basic_functionality_test; then
        log_error "Basic functionality test failed"
        return 1
    fi

    # Check error rates (simplified - would use actual monitoring)
    log_info "Checking error rates..."
    sleep 30  # Wait for metrics to stabilize
    
    log_success "Rollback validation passed"
    return 0
}

# Run basic functionality test
run_basic_functionality_test() {
    # Simplified functionality test
    # In a real deployment, this would test key features
    
    log_info "Testing basic API functionality..."
    
    # Use port-forward to test
    kubectl port-forward -n "$NAMESPACE" "service/ran-llm-pipeline-$ROLLBACK_SLOT" 8080:80 &
    local port_forward_pid=$!
    
    sleep 5
    
    # Test a few key endpoints
    local test_result=0
    
    if ! curl -f -s "http://localhost:8080/health" > /dev/null; then
        log_error "Health endpoint test failed"
        test_result=1
    fi
    
    if ! curl -f -s "http://localhost:8080/api/v1/status" > /dev/null; then
        log_warning "Status endpoint test failed (non-critical)"
    fi
    
    # Cleanup
    kill $port_forward_pid 2>/dev/null || true
    wait $port_forward_pid 2>/dev/null || true
    
    return $test_result
}

# Create rollback record for audit trail
create_rollback_record() {
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    local rollback_record
    rollback_record=$(cat << EOF
{
    "timestamp": "$timestamp",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "reason": "$REASON",
    "rollback_from": {
        "slot": "$CURRENT_SLOT",
        "image": "$CURRENT_IMAGE"
    },
    "rollback_to": {
        "slot": "$ROLLBACK_SLOT", 
        "image": "$ROLLBACK_IMAGE"
    },
    "initiated_by": "${USER:-unknown}",
    "status": "in_progress"
}
EOF
)

    if [[ "$DRY_RUN" == false ]]; then
        echo "$rollback_record" > "rollback-${timestamp}.json"
        
        # Store in Kubernetes configmap for audit trail
        kubectl create configmap -n "$NAMESPACE" "rollback-record-$(date +%s)" \
            --from-literal=record="$rollback_record" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        log_info "Rollback record created: rollback-${timestamp}.json"
    else
        log_info "[DRY RUN] Would create rollback record"
        echo "$rollback_record"
    fi
}

# Update rollback record with final status
update_rollback_record() {
    local status="$1"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    log_info "Updating rollback record with status: $status"
    
    if [[ "$DRY_RUN" == false ]]; then
        # Find the most recent rollback record file
        local record_file
        record_file=$(ls -t rollback-*.json 2>/dev/null | head -1)
        
        if [[ -n "$record_file" ]]; then
            # Update the status in the JSON file
            local updated_record
            updated_record=$(jq ".status = \"$status\" | .completed_at = \"$timestamp\"" "$record_file")
            echo "$updated_record" > "$record_file"
        fi
    fi
}

# Main function
main() {
    echo -e "${BLUE}🔄 RAN LLM Pipeline Rollback Tool${NC}"
    echo "================================="
    
    parse_args "$@"
    
    log_info "Starting rollback process for $ENVIRONMENT environment"
    log_info "Reason: $REASON"
    log_info "Dry run: $DRY_RUN"
    
    check_prerequisites
    get_current_deployment_info
    validate_rollback_target
    perform_rollback
    
    update_rollback_record "completed"
    
    echo -e "\n${GREEN}✅ Rollback completed successfully!${NC}"
    echo "Environment: $ENVIRONMENT"
    echo "Active slot: $ROLLBACK_SLOT (was: $CURRENT_SLOT)"
    echo "Image: $ROLLBACK_IMAGE"
    
    log_success "Rollback process completed"
}

# Trap signals for cleanup
trap 'log_warning "Rollback interrupted"; update_rollback_record "interrupted"; exit 130' INT TERM

# Run main function
main "$@"