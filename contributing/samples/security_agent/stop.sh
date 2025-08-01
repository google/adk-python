#!/bin/bash

# Stop script for ADK services
# This script checks for and stops all running ADK-related processes

set -e  # Exit on any error

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

# Function to stop a service by PID file
stop_service_by_pid_file() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if is_process_running "$pid"; then
            print_status "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            
            # Wait for process to stop (max 10 seconds)
            local count=0
            while is_process_running "$pid" && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if is_process_running "$pid"; then
                print_warning "$service_name did not stop gracefully, forcing..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            print_success "$service_name stopped"
        else
            print_status "$service_name was not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        print_status "No PID file found for $service_name"
    fi
}

# Function to stop processes by port
stop_processes_by_port() {
    local port=$1
    local service_name=$2
    
    print_status "Checking for processes on port $port ($service_name)..."
    
    # Find PIDs listening on the port
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        print_status "Found processes on port $port: $pids"
        for pid in $pids; do
            if is_process_running "$pid"; then
                print_status "Stopping process $pid on port $port..."
                kill "$pid" 2>/dev/null || true
                
                # Wait for process to stop
                local count=0
                while is_process_running "$pid" && [ $count -lt 5 ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                if is_process_running "$pid"; then
                    print_warning "Process $pid did not stop gracefully, forcing..."
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
        print_success "Stopped all processes on port $port"
    else
        print_status "No processes found on port $port"
    fi
}

# Function to stop processes by name pattern
stop_processes_by_name() {
    local pattern=$1
    local service_name=$2
    
    print_status "Checking for $service_name processes matching: $pattern"
    
    # Find PIDs by process name
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        print_status "Found $service_name processes: $pids"
        for pid in $pids; do
            if is_process_running "$pid"; then
                print_status "Stopping $service_name process $pid..."
                kill "$pid" 2>/dev/null || true
                
                # Wait briefly for graceful shutdown
                sleep 1
                
                if is_process_running "$pid"; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
        print_success "Stopped all $service_name processes"
    else
        print_status "No $service_name processes found"
    fi
}

# Function to clean up Docker containers
cleanup_docker() {
    print_status "Checking for Docker containers..."
    
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        # Check for security-agent container
        if docker ps -a --format '{{.Names}}' | grep -q '^security-agent$'; then
            print_status "Stopping security-agent Docker container..."
            docker stop security-agent >/dev/null 2>&1 || true
            docker rm security-agent >/dev/null 2>&1 || true
            print_success "Docker container cleaned up"
        else
            print_status "No security-agent Docker container found"
        fi
    else
        print_status "Docker not available, skipping container cleanup"
    fi
}

# Main execution
main() {
    echo "🛑 Stopping ADK Services"
    echo "=================================="
    
    # Stop services by PID files
    stop_service_by_pid_file "backend.pid" "Backend"
    stop_service_by_pid_file "frontend.pid" "Frontend"
    stop_service_by_pid_file "adk_web.pid" "ADK Web"
    
    # Stop services by port (in case PID files are missing)
    stop_processes_by_port 8000 "Backend API"
    stop_processes_by_port 8501 "Streamlit Frontend"
    stop_processes_by_port 8080 "ADK Web Interface"
    
    # Stop specific processes by name
    stop_processes_by_name "uvicorn.*backend.main:app" "Uvicorn Backend"
    stop_processes_by_name "streamlit.*enhanced_security_agent_app" "Streamlit"
    stop_processes_by_name "google.adk.cli web" "ADK Web"
    
    # Clean up Docker containers
    cleanup_docker
    
    # Clean up log files if requested
    if [ "$1" == "--clean-logs" ]; then
        print_status "Cleaning up log files..."
        rm -f logs/*.log
        print_success "Log files cleaned"
    fi
    
    echo ""
    print_success "All ADK services have been stopped"
    echo "=================================="
}

# Run main function
main "$@"