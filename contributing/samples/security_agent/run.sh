#!/bin/bash

# Enhanced ADK Run Script with Process Checking
# This script checks for existing processes before starting new ones

set -e  # Exit on any error

echo "🚀 Enhanced ADK - One-Command Deployment"
echo "=================================================================="

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
    print_status "Starting backend server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    mkdir -p logs
    
    # Check if backend module exists
    if [ ! -f "backend/main.py" ]; then
        print_warning "backend/main.py not found, skipping backend start"
        return
    fi
    
    nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    print_success "Backend server started (PID: $BACKEND_PID)"
    print_status "Backend log: logs/backend.log"
}

# Function to start frontend
start_frontend() {
    print_status "Starting frontend server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    mkdir -p logs
    
    # Check for frontend app
    local frontend_app=""
    if [ -f "frontend/enhanced_security_agent_app.py" ]; then
        frontend_app="frontend/enhanced_security_agent_app.py"
    elif [ -f "frontend/app.py" ]; then
        frontend_app="frontend/app.py"
    elif [ -f "app.py" ]; then
        frontend_app="app.py"
    else
        print_warning "No frontend app found, skipping frontend start"
        return
    fi
    
    nohup streamlit run $frontend_app --server.port 8501 > logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    print_success "Frontend server started (PID: $FRONTEND_PID)"
    print_status "Frontend log: logs/frontend.log"
}

# Function to start ADK web interface
start_adk_web() {
    print_status "Starting ADK web interface..."
    mkdir -p logs
    
    # Check if ADK is available
    if ! $PYTHON_CMD -m google.adk.cli --help >/dev/null 2>&1; then
        print_warning "ADK CLI not available, skipping ADK web interface"
        return
    fi
    
    $PYTHON_CMD -m google.adk.cli web --port 8080 > logs/adk_web.log 2>&1 &
    ADK_WEB_PID=$!
    echo $ADK_WEB_PID > adk_web.pid
    print_success "ADK web interface started (PID: $ADK_WEB_PID)"
    print_status "ADK web log: logs/adk_web.log"
}

# Function to verify service is running
verify_service() {
    local url=$1
    local service_name=$2
    local retries=5
    local delay=3

    print_status "Verifying $service_name..."
    for i in $(seq 1 $retries); do
        if curl -s --head $url 2>/dev/null | head -n 1 | grep -E "(200|OK)" > /dev/null; then
            print_success "$service_name is running at $url"
            return 0
        fi
        print_status "Waiting for $service_name... (attempt $i/$retries)"
        sleep $delay
    done
    print_warning "$service_name may not have started correctly"
    return 1
}

# Function to show startup info
show_startup_info() {
    echo ""
    echo "🎉 ADK Services are Ready!"
    echo "===================================================="
    echo ""
    echo "🌐 Access Points:"
    
    if [ -f "backend.pid" ] && is_process_running $(cat backend.pid); then
        echo "   • Backend API: http://localhost:8000"
        echo "   • API Documentation: http://localhost:8000/docs"
    fi
    
    if [ -f "frontend.pid" ] && is_process_running $(cat frontend.pid); then
        echo "   • Frontend (Streamlit): http://localhost:8501"
    fi
    
    if [ -f "adk_web.pid" ] && is_process_running $(cat adk_web.pid); then
        echo "   • ADK Web Interface: http://localhost:8080"
    fi
    
    echo ""
    echo "📋 Management Commands:"
    echo "   • Stop all services: ./stop.sh"
    echo "   • View logs: tail -f logs/*.log"
    echo "   • Clean logs: ./stop.sh --clean-logs"
    echo ""
}

# Main execution
main() {
    # Ensure logs directory exists
    mkdir -p logs

    # Redirect all output of this script to run.log
    exec > >(tee -a logs/run.log) 2>&1
    
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
        start_backend
        start_frontend
        start_adk_web
        
        # Wait a moment for services to start
        sleep 2
        
        # Verify services
        if [ -f "backend.pid" ]; then
            verify_service "http://localhost:8000/health" "Backend" || verify_service "http://localhost:8000" "Backend"
        fi
        
        if [ -f "frontend.pid" ]; then
            verify_service "http://localhost:8501" "Frontend"
        fi
        
        if [ -f "adk_web.pid" ]; then
            verify_service "http://localhost:8080" "ADK Web"
        fi
        
        show_startup_info
    fi
}

# Set up trap to handle script interruption
trap 'echo ""; print_warning "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"