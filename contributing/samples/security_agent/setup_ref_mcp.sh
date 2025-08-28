#!/bin/bash

# REF MCP Server Setup Script
# Based on official ref-tools-mcp documentation

set -e

echo "========================================="
echo "REF Tools MCP Server Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed. Please install Node.js and npm first.${NC}"
    exit 1
fi

# Get or prompt for API key
API_KEY="${REF_API_KEY:-}"
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}⚠️  REF_API_KEY not found in environment${NC}"
    echo ""
    echo "To get your API key:"
    echo "1. Visit https://ref.tools"
    echo "2. Sign up for an account"
    echo "3. Get your API key from the dashboard"
    echo ""
    read -p "Enter your REF API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        API_KEY="$api_key"
        # Add to shell profile
        SHELL_PROFILE="$HOME/.zshrc"
        if [ -f "$HOME/.bashrc" ]; then
            SHELL_PROFILE="$HOME/.bashrc"
        fi
        echo "export REF_API_KEY='$api_key'" >> "$SHELL_PROFILE"
        echo -e "${GREEN}✅ API key saved to $SHELL_PROFILE${NC}"
    else
        API_KEY="YOUR_API_KEY_HERE"
        echo -e "${YELLOW}⚠️  Using placeholder API key. Remember to update it later.${NC}"
    fi
fi

# Test if ref-tools-mcp is accessible
echo ""
echo "📦 Testing ref-tools-mcp availability..."
if npm view ref-tools-mcp@latest version &> /dev/null; then
    VERSION=$(npm view ref-tools-mcp@latest version)
    echo -e "${GREEN}✅ ref-tools-mcp is available (version $VERSION)${NC}"
else
    echo -e "${YELLOW}⚠️  Unable to verify ref-tools-mcp package${NC}"
fi

# Backup existing Claude configuration
USER_CONFIG="$HOME/.claude.json"
if [ -f "$USER_CONFIG" ]; then
    BACKUP_FILE="$USER_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$USER_CONFIG" "$BACKUP_FILE"
    echo -e "${GREEN}✅ Backed up existing configuration to: $BACKUP_FILE${NC}"
fi

# Create updated Claude configuration with ref server
echo ""
echo "📝 Updating Claude configuration..."

cat > "$USER_CONFIG" << EOF
{
  "mcpServers": {
    "ref": {
      "command": "npx",
      "args": ["ref-tools-mcp@latest"],
      "env": {
        "REF_API_KEY": "$API_KEY"
      }
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    },
    "claude-flow": {
      "command": "npx",
      "args": ["@claude-flow/mcp-server@latest"],
      "env": {}
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm@latest"],
      "env": {}
    },
    "ide": {
      "command": "node",
      "args": ["/Users/stuartgano/.vscode/extensions/saoudrizwan.claude-dev-2.0.2/dist/mcp/index.js"],
      "env": {}
    }
  }
}
EOF

echo -e "${GREEN}✅ Updated Claude configuration at: $USER_CONFIG${NC}"

# Create a project-level configuration as well
PROJECT_ROOT="/path/to/your/ADK"
PROJECT_CONFIG="$PROJECT_ROOT/.claude/claude.json"
mkdir -p "$PROJECT_ROOT/.claude"

cat > "$PROJECT_CONFIG" << EOF
{
  "mcpServers": {
    "ref": {
      "command": "npx",
      "args": ["ref-tools-mcp@latest"],
      "env": {
        "REF_API_KEY": "$API_KEY"
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Created project configuration at: $PROJECT_CONFIG${NC}"

# Pre-download the ref-tools-mcp package to speed up first use
echo ""
echo "📥 Pre-downloading ref-tools-mcp package..."
if npx --yes ref-tools-mcp@latest --version 2>/dev/null; then
    echo -e "${GREEN}✅ ref-tools-mcp is ready to use${NC}"
else
    echo -e "${YELLOW}⚠️  Package will be downloaded on first use${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}✅ REF MCP Server Setup Complete!${NC}"
echo "========================================="
echo ""
echo "📋 Configuration Summary:"
echo "  - User config: $USER_CONFIG"
echo "  - Project config: $PROJECT_CONFIG"
if [ "$API_KEY" = "YOUR_API_KEY_HERE" ]; then
    echo -e "  - API Key: ${YELLOW}Not configured (placeholder)${NC}"
else
    echo -e "  - API Key: ${GREEN}Configured${NC}"
fi
echo ""
echo "🚀 Next Steps:"
echo "1. Restart Claude Desktop to load the new configuration"
echo "2. Type /mcp in Claude to verify all servers are connected"
echo "3. The 'ref' server should appear alongside your other MCP servers"
echo ""
if [ "$API_KEY" = "YOUR_API_KEY_HERE" ]; then
    echo -e "${YELLOW}⚠️  Remember to get your API key from https://ref.tools${NC}"
    echo "   Then update it in: $USER_CONFIG"
fi
echo ""
echo "📚 Available REF Tools:"
echo "  • ref_search_documentation - Search technical documentation"
echo "  • ref_read_url - Fetch and convert webpage content to markdown"
echo ""
echo "💡 Example usage in Claude:"
echo '   "Search the React documentation for useEffect hooks"'
echo '   "Read the documentation at https://docs.python.org/3/library/asyncio.html"'