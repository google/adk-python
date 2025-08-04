#!/bin/bash

# Enhanced ADK Run Script with Process Checking
# This script checks for existing processes before starting new ones

set -e  # Exit on any error

echo "🚀 Enhanced ADK - One-Command Deployment"
echo "=================================================================="

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo ""
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h        Show this help message"
    echo "  --docker          Run in Docker mode"
    echo "  --force           Force stop existing processes and restart"
    echo "  --skip-setup      Skip service account and environment setup"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                    # Normal startup with service account checks"
    echo "  ./run.sh --docker           # Run in Docker container"
    echo "  ./run.sh --skip-setup       # Skip service account validation"
    echo "  ./run.sh --force            # Stop existing processes and restart"
    echo ""
    echo "The script will automatically:"
    echo "  • Check and configure service account credentials"
    echo "  • Validate required Google Cloud API permissions"
    echo "  • Enable missing APIs (with user consent)"
    echo "  • Set up Python environment and dependencies"
    echo "  • Start backend, frontend, and ADK web services"
    echo ""
    exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to prompt user for yes/no
prompt_yes_no() {
    local prompt_text="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        prompt_text="$prompt_text (Y/n): "
    else
        prompt_text="$prompt_text (y/N): "
    fi
    
    read -p "$prompt_text" -n 1 -r
    echo ""
    
    if [ "$default" = "y" ]; then
        [[ $REPLY =~ ^[Nn]$ ]] && return 1 || return 0
    else
        [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
    fi
}

# Function to check if gcloud is authenticated and available
check_gcloud_auth() {
    if ! command_exists gcloud; then
        print_warning "gcloud CLI not found. Some features may be limited."
        return 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 >/dev/null; then
        print_warning "gcloud is not authenticated. Some features may be limited."
        return 1
    fi
    
    return 0
}

# Function to load environment variables
load_env_vars() {
    if [ -f ".env" ]; then
        print_status "Loading environment variables from .env file..."
        export $(grep -v '^#' .env | xargs)
        return 0
    elif [ -f ".env.example" ]; then
        print_warning "No .env file found, but .env.example exists"
        if prompt_yes_no "Would you like to copy .env.example to .env and configure it now?"; then
            cp .env.example .env
            print_status "Copied .env.example to .env"
            print_status "Please edit .env file with your configuration and run the script again"
            if command_exists code; then
                code .env
            elif command_exists nano; then
                nano .env
            elif command_exists vi; then
                vi .env
            fi
            exit 0
        fi
        return 1
    else
        print_warning "No .env or .env.example file found"
        return 1
    fi
}

# Function to check service account setup
check_service_account() {
    print_status "Checking service account configuration..."
    
    local has_credentials=false
    local project_id=""
    local service_account_email=""
    
    # Check for service account key file (prefer clearer variable name)
    local sa_key_file="${GOOGLE_SERVICE_ACCOUNT_KEY_FILE:-$GOOGLE_APPLICATION_CREDENTIALS}"
    if [ -n "$sa_key_file" ] && [ -f "$sa_key_file" ]; then
        print_success "Service account key file found: $sa_key_file"
        has_credentials=true
        
        # Extract project ID and service account email from the key file
        if command_exists jq; then
            project_id=$(jq -r '.project_id' "$sa_key_file" 2>/dev/null)
            service_account_email=$(jq -r '.client_email' "$sa_key_file" 2>/dev/null)
        elif command_exists python3; then
            project_id=$(python3 -c "import json; print(json.load(open('$sa_key_file')).get('project_id', ''))" 2>/dev/null)
            service_account_email=$(python3 -c "import json; print(json.load(open('$sa_key_file')).get('client_email', ''))" 2>/dev/null)
        fi
    elif [ -n "$GOOGLE_SERVICE_ACCOUNT_JSON" ]; then
        print_success "Service account JSON found in environment variable"
        has_credentials=true
        
        # Extract info from JSON string
        if command_exists jq; then
            project_id=$(echo "$GOOGLE_SERVICE_ACCOUNT_JSON" | jq -r '.project_id' 2>/dev/null)
            service_account_email=$(echo "$GOOGLE_SERVICE_ACCOUNT_JSON" | jq -r '.client_email' 2>/dev/null)
        elif command_exists python3; then
            project_id=$(python3 -c "import json, os; print(json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON']).get('project_id', ''))" 2>/dev/null)
            service_account_email=$(python3 -c "import json, os; print(json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON']).get('client_email', ''))" 2>/dev/null)
        fi
    else
        print_error "No service account credentials found!"
        offer_service_account_setup
        return 1
    fi
    
    # Use project ID from environment or from service account
    if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
        project_id="$GOOGLE_CLOUD_PROJECT"
    fi
    
    if [ -n "$project_id" ]; then
        print_success "Project ID: $project_id"
        export GOOGLE_CLOUD_PROJECT="$project_id"
    else
        print_warning "No project ID found"
    fi
    
    if [ -n "$service_account_email" ]; then
        print_success "Service Account: $service_account_email"
        
        # Check service account permissions if gcloud is available
        if check_gcloud_auth; then
            check_service_account_permissions "$project_id" "$service_account_email"
        fi
    fi
    
    return 0
}

# Function to offer service account setup
offer_service_account_setup() {
    echo ""
    print_warning "Service Account Setup Required"
    echo "=================================================================="
    echo ""
    echo "The Security Agent requires Google Cloud service account credentials."
    echo "You have the following options:"
    echo ""
    echo "1. Use existing service account key file"
    echo "2. Create a new service account (requires gcloud CLI)"
    echo "3. Exit and set up credentials manually"
    echo ""
    
    read -p "Select an option (1-3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            setup_existing_service_account
            ;;
        2)
            create_new_service_account
            ;;
        3)
            print_status "Please set up service account credentials and run again"
            print_status "See .env.example for detailed instructions"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            offer_service_account_setup
            ;;
    esac
}

# Function to setup existing service account
setup_existing_service_account() {
    echo ""
    print_status "Setting up existing service account..."
    
    read -p "Enter path to service account key file: " sa_key_path
    
    if [ ! -f "$sa_key_path" ]; then
        print_error "File not found: $sa_key_path"
        return 1
    fi
    
    # Validate JSON file
    if ! python3 -c "import json; json.load(open('$sa_key_path'))" 2>/dev/null; then
        print_error "Invalid JSON file: $sa_key_path"
        return 1
    fi
    
    # Add to .env file
    if [ -f ".env" ]; then
        sed -i.bak "s|^GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$sa_key_path|" .env
    else
        echo "GOOGLE_APPLICATION_CREDENTIALS=$sa_key_path" > .env
    fi
    
    export GOOGLE_APPLICATION_CREDENTIALS="$sa_key_path"
    print_success "Service account configured: $sa_key_path"
}

# Function to create new service account
create_new_service_account() {
    if ! check_gcloud_auth; then
        print_error "gcloud CLI is required to create service accounts"
        return 1
    fi
    
    echo ""
    print_status "Creating new service account..."
    
    # Get project ID
    local project_id="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
    if [ -z "$project_id" ]; then
        read -p "Enter your GCP Project ID: " project_id
        export GOOGLE_CLOUD_PROJECT="$project_id"
    fi
    
    local sa_name="security-agent"
    local sa_display_name="Security Agent"
    local sa_email="${sa_name}@${project_id}.iam.gserviceaccount.com"
    local key_file="security-agent-key.json"
    
    print_status "Creating service account: $sa_email"
    
    # Create service account
    if gcloud iam service-accounts create "$sa_name" \
        --display-name="$sa_display_name" \
        --project="$project_id" 2>/dev/null; then
        print_success "Service account created"
    else
        print_warning "Service account may already exist"
    fi
    
    # Create and download key
    if gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$sa_email" \
        --project="$project_id"; then
        print_success "Service account key created: $key_file"
        
        # Add to .env file
        if [ -f ".env" ]; then
            sed -i.bak "s|^GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=./$key_file|" .env
            sed -i.bak "s|^GOOGLE_CLOUD_PROJECT=.*|GOOGLE_CLOUD_PROJECT=$project_id|" .env
        else
            echo "GOOGLE_APPLICATION_CREDENTIALS=./$key_file" > .env
            echo "GOOGLE_CLOUD_PROJECT=$project_id" >> .env
        fi
        
        export GOOGLE_APPLICATION_CREDENTIALS="./$key_file"
        
        # Offer to set up permissions
        setup_service_account_permissions "$project_id" "$sa_email"
    else
        print_error "Failed to create service account key"
        return 1
    fi
}

# Function to check service account permissions
check_service_account_permissions() {
    local project_id="$1"
    local sa_email="$2"
    
    if [ -z "$project_id" ] || [ -z "$sa_email" ]; then
        return 1
    fi
    
    print_status "Checking service account permissions..."
    
    # Define required roles
    local core_roles=(
        "roles/cloudtrace.agent"
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
        "roles/serviceusage.serviceUsageConsumer"
        "roles/resourcemanager.projectViewer"
    )
    
    local security_roles=(
        "roles/iam.securityReviewer"
        "roles/recommender.viewer"
    )
    
    local optional_roles=(
        "roles/securitycenter.findingsViewer"
        "roles/cloudsql.viewer"
        "roles/storage.objectViewer"
    )
    
    local missing_core=()
    local missing_security=()
    local missing_optional=()
    
    # Check each role
    for role in "${core_roles[@]}"; do
        if ! gcloud projects get-iam-policy "$project_id" \
            --flatten="bindings[].members" \
            --format="table(bindings.role)" \
            --filter="bindings.members:$sa_email AND bindings.role:$role" 2>/dev/null | grep -q "$role"; then
            missing_core+=("$role")
        fi
    done
    
    for role in "${security_roles[@]}"; do
        if ! gcloud projects get-iam-policy "$project_id" \
            --flatten="bindings[].members" \
            --format="table(bindings.role)" \
            --filter="bindings.members:$sa_email AND bindings.role:$role" 2>/dev/null | grep -q "$role"; then
            missing_security+=("$role")
        fi
    done
    
    for role in "${optional_roles[@]}"; do
        if ! gcloud projects get-iam-policy "$project_id" \
            --flatten="bindings[].members" \
            --format="table(bindings.role)" \
            --filter="bindings.members:$sa_email AND bindings.role:$role" 2>/dev/null | grep -q "$role"; then
            missing_optional+=("$role")
        fi
    done
    
    # Report results
    if [ ${#missing_core[@]} -eq 0 ] && [ ${#missing_security[@]} -eq 0 ]; then
        print_success "All required permissions are configured"
    else
        print_warning "Missing permissions detected"
        
        if [ ${#missing_core[@]} -gt 0 ]; then
            echo ""
            print_error "Missing REQUIRED core permissions:"
            for role in "${missing_core[@]}"; do
                echo "  • $role"
            done
        fi
        
        if [ ${#missing_security[@]} -gt 0 ]; then
            echo ""
            print_error "Missing REQUIRED security permissions:"
            for role in "${missing_security[@]}"; do
                echo "  • $role"
            done
        fi
        
        if [ ${#missing_optional[@]} -gt 0 ]; then
            echo ""
            print_warning "Missing OPTIONAL permissions:"
            for role in "${missing_optional[@]}"; do
                echo "  • $role"
            done
        fi
        
        echo ""
        if prompt_yes_no "Would you like to add the missing permissions now?" "y"; then
            add_service_account_permissions "$project_id" "$sa_email" "${missing_core[@]}" "${missing_security[@]}" "${missing_optional[@]}"
        fi
    fi
    
    # Check enabled APIs
    check_required_apis "$project_id"
}

# Function to setup service account permissions
setup_service_account_permissions() {
    local project_id="$1"
    local sa_email="$2"
    
    echo ""
    print_status "Setting up service account permissions..."
    echo "The Security Agent requires several IAM roles to function properly."
    echo ""
    
    if prompt_yes_no "Would you like to add the required permissions now?" "y"; then
        add_service_account_permissions "$project_id" "$sa_email"
    else
        print_warning "Permissions not added. The application may not work correctly."
        print_status "You can add permissions later using the commands in .env.example"
    fi
}

# Function to add service account permissions
add_service_account_permissions() {
    local project_id="$1"
    local sa_email="$2"
    shift 2
    local roles_to_add=("$@")
    
    # If no specific roles provided, add all required roles
    if [ ${#roles_to_add[@]} -eq 0 ]; then
        roles_to_add=(
            "roles/cloudtrace.agent"
            "roles/logging.logWriter"
            "roles/monitoring.metricWriter"
            "roles/serviceusage.serviceUsageConsumer"
            "roles/resourcemanager.projectViewer"
            "roles/iam.securityReviewer"
            "roles/recommender.viewer"
        )
        
        # Ask about optional roles
        if prompt_yes_no "Would you like to add optional roles for enhanced features?"; then
            roles_to_add+=(
                "roles/securitycenter.findingsViewer"
                "roles/cloudsql.viewer"
                "roles/storage.objectViewer"
            )
        fi
    fi
    
    print_status "Adding IAM roles to service account..."
    
    for role in "${roles_to_add[@]}"; do
        print_status "Adding role: $role"
        if gcloud projects add-iam-policy-binding "$project_id" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet >/dev/null 2>&1; then
            print_success "✓ Added $role"
        else
            print_warning "✗ Failed to add $role (may already exist or insufficient permissions)"
        fi
    done
    
    print_success "Permission setup completed"
}

# Function to check required APIs
check_required_apis() {
    local project_id="$1"
    
    print_status "Checking required Google Cloud APIs..."
    
    local required_apis=(
        "cloudtrace.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
        "serviceusage.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "recommender.googleapis.com"
    )
    
    local optional_apis=(
        "securitycenter.googleapis.com"
        "sqladmin.googleapis.com"
        "storage.googleapis.com"
        "compute.googleapis.com"
    )
    
    local missing_required=()
    local missing_optional=()
    
    # Check required APIs
    for api in "${required_apis[@]}"; do
        if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" --project="$project_id" 2>/dev/null | grep -q "$api"; then
            missing_required+=("$api")
        fi
    done
    
    # Check optional APIs
    for api in "${optional_apis[@]}"; do
        if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" --project="$project_id" 2>/dev/null | grep -q "$api"; then
            missing_optional+=("$api")
        fi
    done
    
    # Report results
    if [ ${#missing_required[@]} -eq 0 ]; then
        print_success "All required APIs are enabled"
    else
        print_warning "Missing required APIs:"
        for api in "${missing_required[@]}"; do
            echo "  • $api"
        done
        
        if prompt_yes_no "Would you like to enable the missing required APIs now?" "y"; then
            enable_apis "$project_id" "${missing_required[@]}"
        fi
    fi
    
    if [ ${#missing_optional[@]} -gt 0 ]; then
        echo ""
        print_status "Optional APIs (for enhanced features):"
        for api in "${missing_optional[@]}"; do
            echo "  • $api"
        done
        
        if prompt_yes_no "Would you like to enable optional APIs for enhanced features?"; then
            enable_apis "$project_id" "${missing_optional[@]}"
        fi
    fi
}

# Function to enable APIs
enable_apis() {
    local project_id="$1"
    shift
    local apis_to_enable=("$@")
    
    print_status "Enabling Google Cloud APIs..."
    
    for api in "${apis_to_enable[@]}"; do
        print_status "Enabling API: $api"
        if gcloud services enable "$api" --project="$project_id" --quiet >/dev/null 2>&1; then
            print_success "✓ Enabled $api"
        else
            print_warning "✗ Failed to enable $api"
        fi
    done
    
    print_success "API enablement completed"
    print_status "Note: It may take a few minutes for newly enabled APIs to be fully available"
}

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if port is in use
is_port_in_use() {
    local port=$1
    lsof -ti:$port >/dev/null 2>&1
}

# Function to check for existing processes
check_existing_processes() {
    print_status "Checking for existing processes..."
    
    local processes_found=false
    
    # Check PID files
    if [ -f "backend.pid" ]; then
        local pid=$(cat backend.pid)
        if is_process_running "$pid"; then
            print_warning "Backend is already running (PID: $pid)"
            processes_found=true
        fi
    fi
    
    if [ -f "frontend.pid" ]; then
        local pid=$(cat frontend.pid)
        if is_process_running "$pid"; then
            print_warning "Frontend is already running (PID: $pid)"
            processes_found=true
        fi
    fi
    
    if [ -f "adk_web.pid" ]; then
        local pid=$(cat adk_web.pid)
        if is_process_running "$pid"; then
            print_warning "ADK Web is already running (PID: $pid)"
            processes_found=true
        fi
    fi
    
    # Check ports
    if is_port_in_use 8000; then
        print_warning "Port 8000 is already in use (Backend API)"
        processes_found=true
    fi
    
    if is_port_in_use 8501; then
        print_warning "Port 8501 is already in use (Streamlit Frontend)"
        processes_found=true
    fi
    
    if is_port_in_use 8080; then
        print_warning "Port 8080 is already in use (ADK Web Interface)"
        processes_found=true
    fi
    
    if [ "$processes_found" = true ]; then
        echo ""
        print_warning "Existing processes detected!"
        echo ""
        echo "Options:"
        echo "  1. Run './stop.sh' to stop all existing processes"
        echo "  2. Run './run.sh --force' to stop existing processes and start fresh"
        echo "  3. Cancel this operation (Ctrl+C)"
        echo ""
        
        if [ "$1" != "--force" ]; then
            read -p "Would you like to stop existing processes and continue? (y/N): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "Operation cancelled"
                exit 0
            fi
        fi
        
        print_status "Stopping existing processes..."
        ./stop.sh
        echo ""
    else
        print_success "No existing processes found"
    fi
}

# --- Docker Workflow ---

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    if ! command_exists docker; then
        print_error "Docker is not installed"
        print_status "Please install Docker from https://www.docker.com/get-started"
        exit 1
    fi

    if ! docker info > /dev/null 2>&1; then
        print_error "Docker daemon is not running"
        print_status "Please start the Docker daemon and try again"
        exit 1
    fi
    print_success "Docker is available and running"
}

# Function to build the Docker image
build_docker_image() {
    print_status "Building Docker image..."
    docker build -t security-agent .
    print_success "Docker image built successfully"
}

# Function to run the Docker container
run_docker_container() {
    print_status "Running Docker container..."
    
    # Stop and remove existing container if it exists
    docker stop security-agent 2>/dev/null || true
    docker rm security-agent 2>/dev/null || true
    
    docker run -p 8000:8000 -p 8501:8501 -d --name security-agent security-agent
    print_success "Docker container is running"
}

# --- Local Workflow ---

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
        PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION is compatible"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command_exists python; then
        PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
        PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION is compatible"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.8+ is not installed"
        print_status "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
}

# Function to check and install pip
check_pip() {
    print_status "Checking pip installation..."
    if ! command_exists pip3 && ! command_exists pip; then
        print_error "pip is not installed"
        print_status "Installing pip..."
        if command_exists curl; then
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
            $PYTHON_CMD get-pip.py --user
            rm get-pip.py
        else
            print_error "curl is required to install pip"
            exit 1
        fi
    fi
    
    if command_exists pip3; then
        PIP_CMD="pip3"
    else
        PIP_CMD="pip"
    fi
    print_success "pip is available"
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    $PIP_CMD install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt
        print_success "Main dependencies installed"
    else
        print_warning "No requirements.txt found in root directory"
    fi
    
    if [ -f "backend/requirements.txt" ]; then
        print_status "Installing backend dependencies..."
        $PIP_CMD install -r backend/requirements.txt
        print_success "Backend dependencies installed"
    else
        print_warning "No backend/requirements.txt found"
    fi
}

# Function to start backend
start_backend() {
    print_status "🔧 Starting backend server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    mkdir -p logs
    
    # Check if backend module exists
    if [ ! -f "backend/main.py" ]; then
        print_warning "backend/main.py not found, skipping backend start"
        return
    fi
    
    # Show what we're doing
    print_status "   • FastAPI backend on port 8000"
    print_status "   • Service account authentication enabled"
    print_status "   • Cloud Trace integration active"
    
    (cd backend && nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &)
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    print_success "✅ Backend server started (PID: $BACKEND_PID)"
    print_status "📄 Backend log: logs/backend.log"
}

# Function to start frontend
start_frontend() {
    print_status "🎨 Starting frontend server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    mkdir -p logs
    
    # Check for frontend app
    local frontend_app=""
    if [ -f "frontend/enhanced_security_agent_app.py" ]; then
        frontend_app="frontend/enhanced_security_agent_app.py"
        print_status "   • Enhanced Security Agent UI"
    elif [ -f "frontend/app.py" ]; then
        frontend_app="frontend/app.py"
        print_status "   • Security Agent UI"
    elif [ -f "app.py" ]; then
        frontend_app="app.py"
        print_status "   • Basic UI"
    else
        print_warning "No frontend app found, skipping frontend start"
        return
    fi
    
    print_status "   • Streamlit server on port 8501"
    print_status "   • Backend connection: http://localhost:8000"
    
    nohup streamlit run $frontend_app --server.port 8501 > logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    print_success "✅ Frontend server started (PID: $FRONTEND_PID)"
    print_status "📄 Frontend log: logs/frontend.log"
}

# Function to start ADK web interface
start_adk_web() {
    print_status "🚀 Starting ADK web interface..."
    mkdir -p logs
    
    # Check if ADK is available
    if ! $PYTHON_CMD -m google.adk.cli --help >/dev/null 2>&1; then
        print_warning "ADK CLI not available, skipping ADK web interface"
        print_status "   • Install ADK CLI for enhanced agent features"
        return
    fi
    
    print_status "   • Native ADK interface on port 8080"
    print_status "   • Agent chat and evaluation tools"
    
    $PYTHON_CMD -m google.adk.cli web --port 8080 > logs/adk_web.log 2>&1 &
    ADK_WEB_PID=$!
    echo $ADK_WEB_PID > adk_web.pid
    print_success "✅ ADK web interface started (PID: $ADK_WEB_PID)"
    print_status "📄 ADK web log: logs/adk_web.log"
}

# Function to show progress spinner
show_spinner() {
    local pid=$1
    local message=$2
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    
    while kill -0 $pid 2>/dev/null; do
        local temp=${spinstr#?}
        printf "\r${BLUE}[INFO]${NC} %s %c" "$message" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r"
}

# Function to wait for backend with progress indicator
wait_for_backend() {
    local retries=30
    local delay=2
    local url="http://localhost:8000/health"
    
    print_status "🔍 Waiting for backend to be ready..."
    
    for i in $(seq 1 $retries); do
        # Show progress bar
        local progress=$((i * 100 / retries))
        local filled=$((progress / 5))
        local empty=$((20 - filled))
        
        printf "\r${BLUE}[INFO]${NC} Backend startup: ["
        printf "%*s" $filled '' | tr ' ' '█'
        printf "%*s" $empty '' | tr ' ' '░'
        printf "] %d%% (attempt %d/%d)" $progress $i $retries
        
        # Check if backend is responding
        if curl -s --max-time 3 "$url" 2>/dev/null | grep -q "healthy\|success\|ok" >/dev/null 2>&1; then
            printf "\n"
            print_success "✅ Backend is ready and responding"
            return 0
        fi
        
        # Also check if backend process is still running
        if [ -f "backend.pid" ]; then
            local backend_pid=$(cat backend.pid)
            if ! kill -0 "$backend_pid" 2>/dev/null; then
                printf "\n"
                print_error "❌ Backend process died during startup"
                print_status "Check logs/backend.log for details"
                return 1
            fi
        fi
        
        sleep $delay
    done
    
    printf "\n"
    print_warning "⚠️ Backend may not have started correctly after ${retries} attempts"
    print_status "You can check the backend status at: $url"
    print_status "Backend logs: tail -f logs/backend.log"
    return 1
}

# Function to verify service is running
verify_service() {
    local url=$1
    local service_name=$2
    local retries=10
    local delay=2

    print_status "🔍 Verifying $service_name..."
    
    for i in $(seq 1 $retries); do
        # Show progress
        printf "\r${BLUE}[INFO]${NC} Checking $service_name... (attempt %d/%d)" $i $retries
        
        if curl -s --max-time 3 $url 2>/dev/null | grep -q "healthy\|running\|OK\|Streamlit\|<title>" >/dev/null 2>&1; then
            printf "\n"
            print_success "✅ $service_name is running at $url"
            return 0
        fi
        
        sleep $delay
    done
    
    printf "\n"
    print_warning "⚠️ $service_name may not have started correctly"
    print_status "URL: $url"
    return 1
}

# Function to show startup info
show_startup_info() {
    echo ""
    echo "🎉 Enhanced Security Agent is Ready!"
    echo "===================================================="
    echo ""
    
    # Show service account status
    if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "🔐 Authentication: Service Account Key"
        if command_exists python3; then
            local sa_email=$(python3 -c "import json; print(json.load(open('$GOOGLE_APPLICATION_CREDENTIALS')).get('client_email', 'Unknown'))" 2>/dev/null)
            if [ -n "$sa_email" ] && [ "$sa_email" != "Unknown" ]; then
                echo "   • Service Account: $sa_email"
            fi
        fi
    elif [ -n "$GOOGLE_SERVICE_ACCOUNT_JSON" ]; then
        echo "🔐 Authentication: Service Account JSON (Environment)"
    else
        echo "🔐 Authentication: Service Account (Method Unknown)"
    fi
    
    if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
        echo "   • Project ID: $GOOGLE_CLOUD_PROJECT"
    fi
    
    echo ""
    echo "🌐 Access Points:"
    
    if [ -f "backend.pid" ] && is_process_running $(cat backend.pid); then
        echo "   • Backend API: http://localhost:8000"
        echo "   • API Documentation: http://localhost:8000/docs"
        echo "   • Health Check: http://localhost:8000/health"
    fi
    
    if [ -f "frontend.pid" ] && is_process_running $(cat frontend.pid); then
        echo "   • Frontend (Streamlit): http://localhost:8501"
    fi
    
    if [ -f "adk_web.pid" ] && is_process_running $(cat adk_web.pid); then
        echo "   • ADK Web Interface: http://localhost:8080"
    fi
    
    echo ""
    echo "🔧 Key Features:"
    echo "   • Enhanced API Services Display with Risk Analysis"
    echo "   • Service Account Authentication for Production Use"
    echo "   • Cloud Trace Integration for Performance Monitoring"
    echo "   • Security Evaluation with IAM Analysis"
    echo "   • Compliance Checking and Threat Intelligence"
    
    echo ""
    echo "📋 Management Commands:"
    echo "   • Stop all services: ./stop.sh"
    echo "   • View logs: tail -f logs/*.log"
    echo "   • Clean logs: ./stop.sh --clean-logs"
    echo "   • Restart with setup checks: ./run.sh"
    echo "   • Skip setup checks: ./run.sh --skip-setup"
    echo ""
}

# Main execution
main() {
    # Ensure logs directory exists
    mkdir -p logs

    # Redirect all output of this script to run.log
    exec > >(tee -a logs/run.log) 2>&1
    
    # Load environment variables (unless skipping setup)
    if [ "$1" != "--skip-setup" ] && [ "$2" != "--skip-setup" ]; then
        load_env_vars
        
        # Check service account setup
        echo ""
        print_status "🔐 Service Account Setup Check"
        echo "=================================================================="
        if ! check_service_account; then
            print_error "Service account setup failed or was cancelled"
            print_status "You can:"
            print_status "  • Run './run.sh --skip-setup' to continue without service account checks"
            print_status "  • Set up credentials manually and run again"
            exit 1
        fi
        echo ""
    else
        print_warning "Skipping service account setup checks"
        # Still try to load basic env vars
        if [ -f ".env" ]; then
            export $(grep -v '^#' .env | xargs) 2>/dev/null || true
        fi
    fi
    
    # Check for existing processes
    check_existing_processes "$1"
    
    if [ "$1" == "--docker" ] || [ "$2" == "--docker" ]; then
        check_docker
        build_docker_image
        run_docker_container
        show_startup_info
    else
        check_python_version
        check_pip
        create_venv
        install_dependencies
        
        echo ""
        print_status "🚀 Starting Services"
        echo "=================================================================="
        
        # Start backend first
        start_backend
        
        # Wait for backend to be ready before starting frontend
        if [ -f "backend.pid" ]; then
            if wait_for_backend; then
                print_success "✅ Backend is ready - starting frontend..."
                sleep 1
            else
                print_warning "⚠️ Backend may not be fully ready, but continuing with frontend startup..."
            fi
        else
            print_warning "⚠️ Backend PID file not found, continuing anyway..."
        fi
        
        # Start frontend after backend is ready
        start_frontend
        
        # Start ADK web interface
        start_adk_web
        
        echo ""
        print_status "🔍 Final Service Verification"
        echo "=================================================================="
        
        # Verify all services with improved checking
        if [ -f "backend.pid" ]; then
            verify_service "http://localhost:8000/health" "Backend Health" || verify_service "http://localhost:8000" "Backend Root"
        fi
        
        if [ -f "frontend.pid" ]; then
            verify_service "http://localhost:8501" "Frontend (Streamlit)"
        fi
        
        if [ -f "adk_web.pid" ]; then
            verify_service "http://localhost:8080" "ADK Web Interface"
        fi
        
        show_startup_info
    fi
}

# Set up trap to handle script interruption
trap 'echo ""; print_warning "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"