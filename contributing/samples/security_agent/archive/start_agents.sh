#!/bin/bash
# Start Claude Code agents with auto-watching for file changes

echo "🚀 Starting Claude Code Agents with Auto-Watch..."
echo "================================================"

# Start all agents with auto-reload and file watching
# --auto: Enable automatic reloading on file changes
# --strict: Enforce strict mode for error handling
# --confirm: Auto-confirm actions without prompting
# --watch: Watch for file changes in the project

# Option 1: Start all agents with file watching
claude-code agents start-all \
    --auto \
    --strict \
    --confirm \
    --watch "**/*.py" \
    --watch "**/*.yaml" \
    --watch "**/*.json" \
    --reload-on-change

# Option 2: Start specific agents for different parts
# Backend watcher
claude-code agent start backend-dev \
    --auto \
    --watch "backend/**/*.py" \
    --watch "*.env" \
    --command "python run_backend.py" \
    --reload-on-change &

# Frontend watcher  
claude-code agent start frontend-dev \
    --auto \
    --watch "frontend/**/*.py" \
    --watch "agents/**/*.py" \
    --command "python run_frontend.py" \
    --reload-on-change &

# API watcher
claude-code agent start api-interface-debugger \
    --auto \
    --watch "backend/api/**/*.py" \
    --strict \
    --confirm &

# Code analyzer for continuous monitoring
claude-code agent start code-analyzer \
    --auto \
    --watch "**/*.py" \
    --continuous \
    --report-issues &

echo ""
echo "✅ Agents started with auto-watch enabled!"
echo ""
echo "Agents are monitoring:"
echo "  📁 Backend files → Auto-restart on changes"
echo "  📁 Frontend files → Auto-reload on changes"
echo "  📁 API files → Debug and validate changes"
echo "  📁 All Python files → Continuous code analysis"
echo ""
echo "Press Ctrl+C to stop all agents"

# Wait for interrupt
wait