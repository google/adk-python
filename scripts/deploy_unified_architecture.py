#!/usr/bin/env python3
"""
Deployment Script for Unified ADK Architecture
Automated deployment of the improved backend, frontend, and API integration.
"""

import os
import sys
import subprocess
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ADKDeploymentManager:
    """Manages the deployment of the unified ADK architecture."""
    
    def __init__(self, project_root: Path, environment: str = "development"):
        self.project_root = project_root
        self.environment = environment
        self.src_dir = project_root / "src"
        self.legacy_dir = project_root / "contributing" / "samples" / "security_agent"
        
        # Deployment configuration
        self.config = {
            "backend": {
                "port": 8000,
                "host": "0.0.0.0",
                "workers": 1 if environment == "development" else 4
            },
            "frontend": {
                "port": 8501,
                "host": "0.0.0.0"
            }
        }
    
    def deploy_full_stack(self) -> bool:
        """Deploy the complete unified architecture."""
        try:
            logger.info("🚀 Starting full stack deployment...")
            
            # Backup existing code
            if not self._backup_existing_code():
                return False
            
            # Deploy backend
            if not self._deploy_backend():
                return False
            
            # Deploy frontend
            if not self._deploy_frontend():
                return False
            
            # Setup configuration
            if not self._setup_configuration():
                return False
            
            # Run tests
            if not self._run_deployment_tests():
                return False
            
            logger.info("✅ Full stack deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def _backup_existing_code(self) -> bool:
        """Backup existing codebase."""
        try:
            logger.info("📦 Backing up existing codebase...")
            
            backup_dir = self.project_root / "backup" / "pre-migration"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup legacy backend
            if self.legacy_dir.exists():
                shutil.copytree(
                    self.legacy_dir / "backend",
                    backup_dir / "legacy_backend",
                    dirs_exist_ok=True
                )
                
                shutil.copytree(
                    self.legacy_dir / "frontend",
                    backup_dir / "legacy_frontend", 
                    dirs_exist_ok=True
                )
            
            # Backup old GCP API Explorer
            old_explorer = self.project_root / "gcp_api_explorer"
            if old_explorer.exists():
                shutil.copytree(
                    old_explorer,
                    backup_dir / "old_gcp_api_explorer",
                    dirs_exist_ok=True
                )
            
            logger.info("✅ Backup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def _deploy_backend(self) -> bool:
        """Deploy the unified backend."""
        try:
            logger.info("🔧 Deploying unified backend...")
            
            backend_src = self.src_dir / "backend"
            backend_dest = self.project_root / "backend"
            
            # Ensure source exists
            if not backend_src.exists():
                logger.error(f"❌ Backend source not found: {backend_src}")
                return False
            
            # Remove old backend if exists
            if backend_dest.exists():
                shutil.rmtree(backend_dest)
            
            # Copy new backend
            shutil.copytree(backend_src, backend_dest)
            
            # Install dependencies
            requirements_file = backend_dest / "requirements.txt"
            if not requirements_file.exists():
                self._create_backend_requirements(requirements_file)
            
            self._install_python_dependencies(backend_dest)
            
            # Create startup script
            self._create_backend_startup_script(backend_dest)
            
            logger.info("✅ Backend deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backend deployment failed: {e}")
            return False
    
    def _deploy_frontend(self) -> bool:
        """Deploy the unified frontend."""
        try:
            logger.info("🎨 Deploying unified frontend...")
            
            frontend_src = self.src_dir / "frontend"
            frontend_dest = self.project_root / "frontend"
            
            # Ensure source exists
            if not frontend_src.exists():
                logger.error(f"❌ Frontend source not found: {frontend_src}")
                return False
            
            # Remove old frontend if exists
            if frontend_dest.exists():
                shutil.rmtree(frontend_dest)
            
            # Copy new frontend
            shutil.copytree(frontend_src, frontend_dest)
            
            # Install dependencies
            requirements_file = frontend_dest / "requirements.txt"
            if not requirements_file.exists():
                self._create_frontend_requirements(requirements_file)
            
            self._install_python_dependencies(frontend_dest)
            
            # Create startup script
            self._create_frontend_startup_script(frontend_dest)
            
            logger.info("✅ Frontend deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend deployment failed: {e}")
            return False
    
    def _setup_configuration(self) -> bool:
        """Setup deployment configuration."""
        try:
            logger.info("⚙️ Setting up configuration...")
            
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Create environment configuration
            env_config = {
                "environment": self.environment,
                "backend_url": f"http://localhost:{self.config['backend']['port']}",
                "frontend_url": f"http://localhost:{self.config['frontend']['port']}",
                "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
                "log_level": "DEBUG" if self.environment == "development" else "INFO"
            }
            
            with open(config_dir / f"{self.environment}.yml", 'w') as f:
                yaml.dump(env_config, f, default_flow_style=False)
            
            # Create docker-compose file for easy deployment
            self._create_docker_compose_file()
            
            # Create environment file
            self._create_env_file()
            
            logger.info("✅ Configuration setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration setup failed: {e}")
            return False
    
    def _create_backend_requirements(self, requirements_file: Path):
        """Create backend requirements.txt."""
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0",
            "google-cloud-resource-manager>=1.10.0",
            "google-cloud-compute>=1.15.0",
            "google-cloud-storage>=2.10.0",
            "google-cloud-iam>=2.12.0",
            "google-api-python-client>=2.100.0",
            "google-auth>=2.23.0",
            "httpx>=0.25.0",
            "python-multipart>=0.0.6",
            "python-dotenv>=1.0.0",
            "tenacity>=8.2.0"
        ]
        
        requirements_file.write_text('\n'.join(requirements))
    
    def _create_frontend_requirements(self, requirements_file: Path):
        """Create frontend requirements.txt."""
        requirements = [
            "streamlit>=1.28.0",
            "pandas>=2.1.0",
            "plotly>=5.17.0",
            "requests>=2.31.0",
            "python-dotenv>=1.0.0"
        ]
        
        requirements_file.write_text('\n'.join(requirements))
    
    def _install_python_dependencies(self, directory: Path):
        """Install Python dependencies."""
        try:
            logger.info(f"📦 Installing dependencies in {directory}")
            
            # Check if venv exists, create if not
            venv_path = directory / "venv"
            if not venv_path.exists():
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], check=True)
            
            # Install requirements
            pip_path = venv_path / "bin" / "pip"
            if not pip_path.exists():
                pip_path = venv_path / "Scripts" / "pip.exe"  # Windows
            
            if pip_path.exists():
                subprocess.run([
                    str(pip_path), "install", "-r", "requirements.txt"
                ], cwd=directory, check=True)
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Dependency installation failed: {e}")
    
    def _create_backend_startup_script(self, backend_dir: Path):
        """Create backend startup script."""
        script_content = f"""#!/bin/bash
# ADK Backend Startup Script

cd "{backend_dir}"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Set environment variables
export HOST={self.config['backend']['host']}
export PORT={self.config['backend']['port']}
export ENVIRONMENT={self.environment}

# Start the backend
echo "🚀 Starting ADK Security Agent Backend..."
echo "   • Host: $HOST"
echo "   • Port: $PORT"
echo "   • Environment: $ENVIRONMENT"
echo ""

python main_unified.py
"""
        
        script_path = backend_dir / "start_backend.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        # Windows batch file
        batch_content = f"""@echo off
cd /d "{backend_dir}"

if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)

set HOST={self.config['backend']['host']}
set PORT={self.config['backend']['port']}
set ENVIRONMENT={self.environment}

echo 🚀 Starting ADK Security Agent Backend...
echo    • Host: %HOST%
echo    • Port: %PORT%
echo    • Environment: %ENVIRONMENT%
echo.

python main_unified.py
"""
        
        batch_path = backend_dir / "start_backend.bat"
        batch_path.write_text(batch_content)
    
    def _create_frontend_startup_script(self, frontend_dir: Path):
        """Create frontend startup script."""
        script_content = f"""#!/bin/bash
# ADK Frontend Startup Script

cd "{frontend_dir}"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Set environment variables
export BACKEND_URL=http://localhost:{self.config['backend']['port']}

# Start the frontend
echo "🎨 Starting ADK Security Agent Frontend..."
echo "   • Backend URL: $BACKEND_URL"
echo "   • Frontend Port: {self.config['frontend']['port']}"
echo ""

streamlit run main_dashboard.py --server.port {self.config['frontend']['port']} --server.address {self.config['frontend']['host']}
"""
        
        script_path = frontend_dir / "start_frontend.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        # Windows batch file
        batch_content = f"""@echo off
cd /d "{frontend_dir}"

if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)

set BACKEND_URL=http://localhost:{self.config['backend']['port']}

echo 🎨 Starting ADK Security Agent Frontend...
echo    • Backend URL: %BACKEND_URL%
echo    • Frontend Port: {self.config['frontend']['port']}
echo.

streamlit run main_dashboard.py --server.port {self.config['frontend']['port']} --server.address {self.config['frontend']['host']}
"""
        
        batch_path = frontend_dir / "start_frontend.bat"
        batch_path.write_text(batch_content)
    
    def _create_docker_compose_file(self):
        """Create docker-compose.yml for containerized deployment."""
        compose_content = {
            "version": "3.8",
            "services": {
                "backend": {
                    "build": "./backend",
                    "ports": [f"{self.config['backend']['port']}:{self.config['backend']['port']}"],
                    "environment": {
                        "HOST": "0.0.0.0",
                        "PORT": self.config['backend']['port'],
                        "ENVIRONMENT": self.environment,
                        "GOOGLE_CLOUD_PROJECT": "${GOOGLE_CLOUD_PROJECT}",
                        "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
                    },
                    "volumes": [
                        "${GOOGLE_APPLICATION_CREDENTIALS}:/app/credentials.json:ro"
                    ]
                },
                "frontend": {
                    "build": "./frontend",
                    "ports": [f"{self.config['frontend']['port']}:{self.config['frontend']['port']}"],
                    "environment": {
                        "BACKEND_URL": f"http://backend:{self.config['backend']['port']}"
                    },
                    "depends_on": ["backend"]
                }
            }
        }
        
        with open(self.project_root / "docker-compose.yml", 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
    
    def _create_env_file(self):
        """Create .env file template."""
        env_content = f"""# ADK Security Agent Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Backend Configuration
BACKEND_HOST={self.config['backend']['host']}
BACKEND_PORT={self.config['backend']['port']}

# Frontend Configuration
FRONTEND_HOST={self.config['frontend']['host']}
FRONTEND_PORT={self.config['frontend']['port']}

# Environment
ENVIRONMENT={self.environment}
LOG_LEVEL=INFO

# Optional: Service-specific configuration
CACHE_TTL=3600
MAX_RETRIES=3
REQUEST_TIMEOUT=30
"""
        
        env_file = self.project_root / ".env.example"
        env_file.write_text(env_content)
    
    def _run_deployment_tests(self) -> bool:
        """Run basic deployment validation tests."""
        try:
            logger.info("🧪 Running deployment tests...")
            
            # Test backend startup
            backend_test = self._test_backend_startup()
            if not backend_test:
                return False
            
            # Test frontend startup
            frontend_test = self._test_frontend_startup()
            if not frontend_test:
                return False
            
            logger.info("✅ All deployment tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment tests failed: {e}")
            return False
    
    def _test_backend_startup(self) -> bool:
        """Test backend startup."""
        try:
            logger.info("Testing backend startup...")
            
            backend_dir = self.project_root / "backend"
            main_file = backend_dir / "main_unified.py"
            
            if not main_file.exists():
                logger.error("❌ Backend main file not found")
                return False
            
            # Basic syntax check
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(main_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Backend syntax check failed: {result.stderr}")
                return False
            
            logger.info("✅ Backend startup test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backend test failed: {e}")
            return False
    
    def _test_frontend_startup(self) -> bool:
        """Test frontend startup."""
        try:
            logger.info("Testing frontend startup...")
            
            frontend_dir = self.project_root / "frontend"
            main_file = frontend_dir / "main_dashboard.py"
            
            if not main_file.exists():
                logger.error("❌ Frontend main file not found")
                return False
            
            # Basic syntax check
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(main_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Frontend syntax check failed: {result.stderr}")
                return False
            
            logger.info("✅ Frontend startup test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend test failed: {e}")
            return False
    
    def create_deployment_summary(self):
        """Create deployment summary report."""
        summary = {
            "deployment_info": {
                "timestamp": "$(date)",
                "environment": self.environment,
                "project_root": str(self.project_root),
                "backend_port": self.config['backend']['port'],
                "frontend_port": self.config['frontend']['port']
            },
            "deployed_components": [
                "Unified Backend API",
                "Unified Frontend UI", 
                "GCP Client Service",
                "ADK Evaluator Service",
                "API Explorer Service",
                "Configuration Management"
            ],
            "startup_commands": {
                "backend": "./backend/start_backend.sh",
                "frontend": "./frontend/start_frontend.sh", 
                "docker": "docker-compose up"
            },
            "urls": {
                "backend_api": f"http://localhost:{self.config['backend']['port']}",
                "api_docs": f"http://localhost:{self.config['backend']['port']}/docs",
                "frontend": f"http://localhost:{self.config['frontend']['port']}",
                "health_check": f"http://localhost:{self.config['backend']['port']}/health"
            }
        }
        
        with open(self.project_root / "DEPLOYMENT_SUMMARY.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme_content = f"""# ADK Security Agent - Unified Architecture

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud SDK
- Valid GCP project with appropriate permissions

### Environment Setup
1. Copy `.env.example` to `.env`
2. Update `GOOGLE_CLOUD_PROJECT` with your project ID
3. Set `GOOGLE_APPLICATION_CREDENTIALS` path

### Starting the Application

#### Option 1: Local Development
```bash
# Start backend
cd backend && ./start_backend.sh

# Start frontend (new terminal)
cd frontend && ./start_frontend.sh
```

#### Option 2: Docker
```bash
docker-compose up
```

### Access Points
- **Frontend**: http://localhost:{self.config['frontend']['port']}
- **Backend API**: http://localhost:{self.config['backend']['port']}
- **API Docs**: http://localhost:{self.config['backend']['port']}/docs
- **Health Check**: http://localhost:{self.config['backend']['port']}/health

## 📚 Documentation
- [Architecture Guide](docs/IMPROVED_ARCHITECTURE.md)
- [Migration Plan](docs/MIGRATION_PLAN.md) 
- [ADK Showcase Guide](docs/GCP_ADK_SHOWCASE_GUIDE.md)

## 🎯 Key Features
- Unified GCP integration
- Dynamic API discovery
- Interactive endpoint testing
- Security evaluation engine
- ADK feature showcase

## 🔧 Troubleshooting
- Ensure GCP credentials are properly configured
- Check that required ports are available
- Verify Google Cloud SDK is installed and authenticated

For detailed troubleshooting, see the documentation files.
"""
        
        with open(self.project_root / "README.md", 'w') as f:
            f.write(readme_content)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy ADK Unified Architecture")
    parser.add_argument(
        "--environment", 
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Only perform backup, don't deploy"
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = ADKDeploymentManager(args.project_root, args.environment)
    
    if args.backup_only:
        logger.info("📦 Performing backup only...")
        success = manager._backup_existing_code()
    else:
        logger.info(f"🚀 Starting deployment to {args.environment} environment...")
        success = manager.deploy_full_stack()
    
    if success:
        manager.create_deployment_summary()
        logger.info("✅ Deployment completed successfully!")
        print(f"""
🎉 ADK Security Agent Deployment Complete!

📁 Project Root: {args.project_root}
🌍 Environment: {args.environment}

🚀 Next Steps:
1. Review the deployment summary: DEPLOYMENT_SUMMARY.json
2. Configure your .env file
3. Start the backend: ./backend/start_backend.sh
4. Start the frontend: ./frontend/start_frontend.sh
5. Visit: http://localhost:{manager.config['frontend']['port']}

📚 Documentation available in the docs/ directory.
        """)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()