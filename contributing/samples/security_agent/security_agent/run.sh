#!/bin/bash

# Enhanced GCP Security Agent - One-Command Deployment
# This script sets up and runs the entire security agent from scratch

set -e  # Exit on any error

echo "🚀 Enhanced GCP Security Agent - One-Command Deployment"
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
    $PIP_CMD install -r requirements.txt
    print_success "Main dependencies installed"
    print_status "Installing backend dependencies..."
    $PIP_CMD install -r backend/requirements.txt
    print_success "Backend dependencies installed"

}

# Function to start backend
start_backend() {
    print_status "Starting backend server..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    mkdir -p logs
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
    nohup python -m streamlit run frontend/enhanced_security_agent_app.py --server.port 8501 > logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    print_success "Frontend server started (PID: $FRONTEND_PID)"
    print_status "Frontend log: logs/frontend.log"
}

# Function to start ADK web interface
start_adk_web() {
    print_status "Starting ADK web interface..."
    mkdir -p logs
    python -m google.adk.cli web --port 8080 > logs/adk_web.log 2>&1 &
    ADK_WEB_PID=$!
    echo $ADK_WEB_PID > adk_web.pid
    print_success "ADK web interface started"
}

# --- Main Execution ---

# Function to show startup info
show_startup_info() {
    echo ""
    echo "🎉 Enhanced GCP Security Agent is Ready!"
    echo "===================================================="
    echo ""
    echo "🌐 Access Points:"
    echo "   • Frontend (Streamlit): http://localhost:8501"
    echo "   • Backend API: http://localhost:8000"
    echo "   • API Documentation: http://localhost:8000/docs"
    echo "   • ADK Web Interface: http://localhost:8080"
    echo ""
    echo "🚀 Quick Start:"
    echo "   1. Open http://localhost:8501 in your browser"
    echo "   2. Use the ADK Chat or explore other features"
    echo "   3. Access native ADK interface at http://localhost:8080"
    echo ""
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [ -f "backend.pid" ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm -f backend.pid
    fi
    if [ -f "frontend.pid" ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm -f frontend.pid
    fi
    if [ -f "adk_web.pid" ]; then
        kill $(cat adk_web.pid) 2>/dev/null || true
        rm -f adk_web.pid
    fi
    print_success "Cleanup complete"
}

# Function to verify logs are being written
verify_logs() {
    print_status "Verifying log files..."
    sleep 3  # Give services time to start writing logs
    
    if [ -s "backend.log" ]; then
        print_success "Backend log is being written ($(wc -l < backend.log) lines)"
    else
        print_warning "Backend log is empty or not created"
    fi
    
    if [ -s "frontend.log" ]; then
        print_success "Frontend log is being written ($(wc -l < frontend.log) lines)"
    else
        print_warning "Frontend log is empty or not created"
    fi
    
    if [ -s "adk_web.log" ]; then
        print_success "ADK web log is being written ($(wc -l < adk_web.log) lines)"
    else
        print_warning "ADK web log is empty or not created"
    fi
}

# Function to verify service is running
verify_service() {
    local url=$1
    local service_name=$2
    local retries=5
    local delay=3

    print_status "Verifying $service_name..."
    for i in $(seq 1 $retries); do
        if curl -s --head $url | head -n 1 | grep "200 OK" > /dev/null; then
            print_success "$service_name is running at $url"
            return 0
        fi
        print_status "Waiting for $service_name... (attempt $i/$retries)"
        sleep $delay
    done
    print_error "$service_name failed to start"
    return 1
}

# Main execution
main() {
    # Ensure logs directory exists
    mkdir -p logs

    # Redirect all output of this script to run.log
    exec > >(tee -a logs/run.log) 2>&1
    
    # Pre-flight check
    print_status "Running pre-flight checks..."
    ./stop.sh

    if [ "$1" == "--docker" ]; then
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
        
        # Verify services
        verify_service "http://localhost:8000/health" "Backend"
        verify_service "http://localhost:8501" "Frontend"
        verify_service "http://localhost:8080" "ADK Web"
        
        show_startup_info
        # The script will now exit, but the services will keep running in the background.
        # Use stop.sh to stop them.
    fi
}

# Run main function
main "$@"
