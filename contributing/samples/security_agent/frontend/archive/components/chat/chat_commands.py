"""Chat Command System for ADK Security Agent.

This module implements a comprehensive chat command system that enables users to
navigate and control the application through natural language commands and
structured chat commands.

Features:
    - Structured command parsing (/security, /iam, /compliance, etc.)
    - Natural language command recognition
    - Context-aware command suggestions
    - Command auto-completion
    - Command history and favorites
    - Integration with ADK delegation patterns

Command Categories:
    - Navigation: /dashboard, /security, /iam, /compliance
    - Analysis: /scan, /analyze, /report
    - Agent Control: /agent, /transfer, /status
    - Session: /history, /clear, /export, /new
    - Help: /help, /commands, /examples

Usage:
    from components.chat.chat_commands import ChatCommandProcessor
    
    processor = ChatCommandProcessor()
    result = processor.process_command("/security scan --project my-project")
"""

import re
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class CommandType(Enum):
    """Types of chat commands."""
    NAVIGATION = "navigation"
    ANALYSIS = "analysis"
    AGENT = "agent"
    SESSION = "session"
    HELP = "help"
    CUSTOM = "custom"


@dataclass
class CommandResult:
    """Result of command processing."""
    success: bool
    message: str
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


@dataclass
class Command:
    """Command definition."""
    name: str
    category: CommandType
    description: str
    usage: str
    examples: List[str]
    aliases: List[str] = None
    parameters: List[str] = None


class ChatCommandProcessor:
    """Processes chat commands and natural language queries."""
    
    def __init__(self):
        self.commands = self._initialize_commands()
        self.command_history = []
        self.favorites = set()
        
    def _initialize_commands(self) -> Dict[str, Command]:
        """Initialize the command registry."""
        commands = {}
        
        # Navigation commands
        nav_commands = [
            Command(
                name="dashboard",
                category=CommandType.NAVIGATION,
                description="Switch to dashboard overview",
                usage="/dashboard",
                examples=["/dashboard", "/dash"],
                aliases=["dash", "home", "overview"]
            ),
            Command(
                name="security",
                category=CommandType.NAVIGATION,
                description="Switch to security analysis context",
                usage="/security [action]",
                examples=["/security", "/security scan", "/security findings"],
                aliases=["sec", "secure"],
                parameters=["scan", "findings", "recommendations"]
            ),
            Command(
                name="iam",
                category=CommandType.NAVIGATION,
                description="Switch to IAM analysis context",
                usage="/iam [action]",
                examples=["/iam", "/iam analyze", "/iam users"],
                aliases=["identity", "access"],
                parameters=["analyze", "users", "policies", "permissions"]
            ),
            Command(
                name="compliance",
                category=CommandType.NAVIGATION,
                description="Switch to compliance context",
                usage="/compliance [framework]",
                examples=["/compliance", "/compliance soc2", "/compliance gdpr"],
                aliases=["comp", "audit"],
                parameters=["soc2", "gdpr", "hipaa", "iso27001", "pci"]
            )
        ]
        
        # Analysis commands
        analysis_commands = [
            Command(
                name="scan",
                category=CommandType.ANALYSIS,
                description="Run security scan",
                usage="/scan [target]",
                examples=["/scan", "/scan project", "/scan resources"],
                parameters=["project", "resources", "iam", "storage"]
            ),
            Command(
                name="analyze",
                category=CommandType.ANALYSIS,
                description="Analyze specific resource or finding",
                usage="/analyze <resource>",
                examples=["/analyze bucket-name", "/analyze iam-policy"],
                parameters=["bucket", "policy", "instance", "finding"]
            ),
            Command(
                name="report",
                category=CommandType.ANALYSIS,
                description="Generate report",
                usage="/report <type>",
                examples=["/report security", "/report compliance"],
                parameters=["security", "compliance", "iam", "summary"]
            )
        ]
        
        # Agent commands
        agent_commands = [
            Command(
                name="agent",
                category=CommandType.AGENT,
                description="Direct message to specific agent",
                usage="/agent <agent_name> <message>",
                examples=["/agent security show findings", "/agent iam list users"],
                parameters=["security", "iam", "compliance", "coordinator"]
            ),
            Command(
                name="agents",
                category=CommandType.AGENT,
                description="Show all available agents",
                usage="/agents",
                examples=["/agents", "/agents status"],
                aliases=["status"]
            ),
            Command(
                name="transfer",
                category=CommandType.AGENT,
                description="Transfer conversation to specific agent",
                usage="/transfer <agent_name>",
                examples=["/transfer security", "/transfer iam"],
                parameters=["security", "iam", "compliance", "coordinator"]
            )
        ]
        
        # Session commands
        session_commands = [
            Command(
                name="history",
                category=CommandType.SESSION,
                description="Show conversation history",
                usage="/history [count]",
                examples=["/history", "/history 10"],
                aliases=["hist"]
            ),
            Command(
                name="clear",
                category=CommandType.SESSION,
                description="Clear current session",
                usage="/clear",
                examples=["/clear"],
                aliases=["clean", "reset"]
            ),
            Command(
                name="export",
                category=CommandType.SESSION,
                description="Export conversation",
                usage="/export [format]",
                examples=["/export", "/export json", "/export pdf"],
                parameters=["json", "pdf", "txt", "csv"]
            ),
            Command(
                name="new",
                category=CommandType.SESSION,
                description="Start new session",
                usage="/new [name]",
                examples=["/new", "/new security-review"],
                aliases=["session"]
            )
        ]
        
        # Help commands
        help_commands = [
            Command(
                name="help",
                category=CommandType.HELP,
                description="Show help information",
                usage="/help [command]",
                examples=["/help", "/help security", "/help commands"],
                aliases=["?", "commands"]
            ),
            Command(
                name="examples",
                category=CommandType.HELP,
                description="Show example queries",
                usage="/examples [category]",
                examples=["/examples", "/examples security"],
                aliases=["sample", "demo"]
            )
        ]
        
        # Compile all commands
        all_commands = nav_commands + analysis_commands + agent_commands + session_commands + help_commands
        
        for cmd in all_commands:
            commands[cmd.name] = cmd
            # Add aliases
            if cmd.aliases:
                for alias in cmd.aliases:
                    commands[alias] = cmd
        
        return commands
    
    def process_input(self, user_input: str) -> CommandResult:
        """Process user input - either command or natural language."""
        user_input = user_input.strip()
        
        # Check if it's a structured command
        if user_input.startswith('/'):
            return self.process_structured_command(user_input)
        
        # Check for natural language commands
        nl_result = self.process_natural_language(user_input)
        if nl_result.success:
            return nl_result
        
        # Not a command, return as regular query
        return CommandResult(
            success=False,
            message="Regular query - pass to ADK agent",
            action="query",
            data={"query": user_input}
        )
    
    def process_structured_command(self, command_str: str) -> CommandResult:
        """Process structured command (starts with /)."""
        # Parse command
        parts = command_str[1:].split()
        if not parts:
            return CommandResult(
                success=False,
                message="Empty command. Type /help for available commands."
            )
        
        command_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Find command
        if command_name not in self.commands:
            similar = self.find_similar_commands(command_name)
            suggestion_text = f" Did you mean: {', '.join(similar)}?" if similar else ""
            return CommandResult(
                success=False,
                message=f"Unknown command: /{command_name}.{suggestion_text}",
                suggestions=[f"/{cmd}" for cmd in similar]
            )
        
        command = self.commands[command_name]
        
        # Add to history
        self.command_history.append({
            'command': command_str,
            'timestamp': datetime.now(),
            'category': command.category.value
        })
        
        # Execute command
        return self.execute_command(command, args)
    
    def process_natural_language(self, query: str) -> CommandResult:
        """Process natural language for command recognition."""
        query_lower = query.lower()
        
        # Navigation patterns
        nav_patterns = {
            r'(show|go to|open|switch to) dashboard': 'dashboard',
            r'(show|go to|open|switch to) security': 'security',
            r'(show|go to|open|switch to) iam': 'iam',
            r'(show|go to|open|switch to) compliance': 'compliance',
        }
        
        # Analysis patterns
        analysis_patterns = {
            r'(run|start|do) (security )?scan': 'scan',
            r'analyze (my )?(.+)': 'analyze',
            r'(generate|create|make) (.+) report': 'report',
        }
        
        # Help patterns
        help_patterns = {
            r'(help|how to|what can|commands|examples)': 'help',
            r'(show|list) (all )?commands': 'help',
        }
        
        # Check patterns
        all_patterns = {**nav_patterns, **analysis_patterns, **help_patterns}
        
        for pattern, command_name in all_patterns.items():
            if re.search(pattern, query_lower):
                if command_name in self.commands:
                    command = self.commands[command_name]
                    
                    # Extract arguments if needed
                    args = self.extract_args_from_natural_language(query, pattern, command_name)
                    
                    return self.execute_command(command, args)
        
        return CommandResult(success=False, message="Not a recognized command pattern")
    
    def execute_command(self, command: Command, args: List[str]) -> CommandResult:
        """Execute a specific command."""
        if command.category == CommandType.NAVIGATION:
            return self.execute_navigation_command(command, args)
        elif command.category == CommandType.ANALYSIS:
            return self.execute_analysis_command(command, args)
        elif command.category == CommandType.AGENT:
            return self.execute_agent_command(command, args)
        elif command.category == CommandType.SESSION:
            return self.execute_session_command(command, args)
        elif command.category == CommandType.HELP:
            return self.execute_help_command(command, args)
        else:
            return CommandResult(
                success=False,
                message=f"Unknown command category: {command.category}"
            )
    
    def execute_navigation_command(self, command: Command, args: List[str]) -> CommandResult:
        """Execute navigation commands."""
        if command.name in ['dashboard', 'dash', 'home', 'overview']:\n            # Switch to dashboard\n            st.session_state.page = 'dashboard'\n            return CommandResult(\n                success=True,\n                message=\"🏠 Switched to Dashboard\",\n                action=\"navigate\",\n                data={\"page\": \"dashboard\"}\n            )\n        \n        elif command.name in ['security', 'sec', 'secure']:\n            # Switch to security context\n            st.session_state.page = 'security'\n            if args:\n                action = args[0]\n                if action == 'scan':\n                    return CommandResult(\n                        success=True,\n                        message=\"🛡️ Starting security scan...\",\n                        action=\"security_scan\",\n                        data={\"page\": \"security\", \"action\": \"scan\"}\n                    )\n                elif action == 'findings':\n                    return CommandResult(\n                        success=True,\n                        message=\"🔍 Showing security findings...\",\n                        action=\"security_findings\",\n                        data={\"page\": \"security\", \"action\": \"findings\"}\n                    )\n            \n            return CommandResult(\n                success=True,\n                message=\"🛡️ Switched to Security Analysis\",\n                action=\"navigate\",\n                data={\"page\": \"security\"}\n            )\n        \n        elif command.name in ['iam', 'identity', 'access']:\n            # Switch to IAM context\n            st.session_state.page = 'iam'\n            if args:\n                action = args[0]\n                if action == 'analyze':\n                    return CommandResult(\n                        success=True,\n                        message=\"🔐 Analyzing IAM configuration...\",\n                        action=\"iam_analyze\",\n                        data={\"page\": \"iam\", \"action\": \"analyze\"}\n                    )\n                elif action == 'users':\n                    return CommandResult(\n                        success=True,\n                        message=\"👥 Listing IAM users...\",\n                        action=\"iam_users\",\n                        data={\"page\": \"iam\", \"action\": \"users\"}\n                    )\n            \n            return CommandResult(\n                success=True,\n                message=\"🔐 Switched to IAM Analysis\",\n                action=\"navigate\",\n                data={\"page\": \"iam\"}\n            )\n        \n        elif command.name in ['compliance', 'comp', 'audit']:\n            # Switch to compliance context\n            st.session_state.page = 'compliance'\n            if args:\n                framework = args[0].upper()\n                return CommandResult(\n                    success=True,\n                    message=f\"📋 Checking {framework} compliance...\",\n                    action=\"compliance_check\",\n                    data={\"page\": \"compliance\", \"framework\": framework}\n                )\n            \n            return CommandResult(\n                success=True,\n                message=\"📋 Switched to Compliance\",\n                action=\"navigate\",\n                data={\"page\": \"compliance\"}\n            )\n        \n        return CommandResult(\n            success=False,\n            message=f\"Navigation command not implemented: {command.name}\"\n        )\n    \n    def execute_analysis_command(self, command: Command, args: List[str]) -> CommandResult:\n        \"\"\"Execute analysis commands.\"\"\"\n        if command.name == 'scan':\n            target = args[0] if args else 'project'\n            return CommandResult(\n                success=True,\n                message=f\"🔍 Starting {target} scan...\",\n                action=\"scan\",\n                data={\"target\": target}\n            )\n        \n        elif command.name == 'analyze':\n            if not args:\n                return CommandResult(\n                    success=False,\n                    message=\"Please specify what to analyze. Usage: /analyze <resource>\"\n                )\n            \n            resource = ' '.join(args)\n            return CommandResult(\n                success=True,\n                message=f\"🔍 Analyzing {resource}...\",\n                action=\"analyze\",\n                data={\"resource\": resource}\n            )\n        \n        elif command.name == 'report':\n            report_type = args[0] if args else 'summary'\n            return CommandResult(\n                success=True,\n                message=f\"📊 Generating {report_type} report...\",\n                action=\"report\",\n                data={\"type\": report_type}\n            )\n        \n        return CommandResult(\n            success=False,\n            message=f\"Analysis command not implemented: {command.name}\"\n        )\n    \n    def execute_agent_command(self, command: Command, args: List[str]) -> CommandResult:\n        \"\"\"Execute agent commands.\"\"\"\n        if command.name == 'agents' or command.name == 'status':\n            return CommandResult(\n                success=True,\n                message=\"🤖 Available ADK Agents:\",\n                action=\"show_agents\",\n                data={\n                    \"agents\": [\n                        {\"name\": \"Coordinator\", \"status\": \"Active\", \"icon\": \"🎯\"},\n                        {\"name\": \"Security\", \"status\": \"Ready\", \"icon\": \"🛡️\"},\n                        {\"name\": \"IAM\", \"status\": \"Ready\", \"icon\": \"🔐\"},\n                        {\"name\": \"Compliance\", \"status\": \"Ready\", \"icon\": \"📋\"}\n                    ]\n                }\n            )\n        \n        elif command.name == 'agent':\n            if len(args) < 2:\n                return CommandResult(\n                    success=False,\n                    message=\"Usage: /agent <agent_name> <message>\"\n                )\n            \n            agent_name = args[0]\n            message = ' '.join(args[1:])\n            \n            return CommandResult(\n                success=True,\n                message=f\"🤖 Routing to {agent_name} agent: {message}\",\n                action=\"direct_agent\",\n                data={\"agent\": agent_name, \"message\": message}\n            )\n        \n        elif command.name == 'transfer':\n            if not args:\n                return CommandResult(\n                    success=False,\n                    message=\"Usage: /transfer <agent_name>\"\n                )\n            \n            agent_name = args[0]\n            return CommandResult(\n                success=True,\n                message=f\"🔄 Transferring conversation to {agent_name} agent\",\n                action=\"transfer_agent\",\n                data={\"agent\": agent_name}\n            )\n        \n        return CommandResult(\n            success=False,\n            message=f\"Agent command not implemented: {command.name}\"\n        )\n    \n    def execute_session_command(self, command: Command, args: List[str]) -> CommandResult:\n        \"\"\"Execute session commands.\"\"\"\n        if command.name in ['history', 'hist']:\n            count = int(args[0]) if args and args[0].isdigit() else 10\n            return CommandResult(\n                success=True,\n                message=f\"📜 Showing last {count} messages\",\n                action=\"show_history\",\n                data={\"count\": count}\n            )\n        \n        elif command.name in ['clear', 'clean', 'reset']:\n            return CommandResult(\n                success=True,\n                message=\"🧹 Clearing current session\",\n                action=\"clear_session\"\n            )\n        \n        elif command.name == 'export':\n            format_type = args[0] if args else 'json'\n            return CommandResult(\n                success=True,\n                message=f\"💾 Exporting conversation as {format_type}\",\n                action=\"export_session\",\n                data={\"format\": format_type}\n            )\n        \n        elif command.name in ['new', 'session']:\n            session_name = args[0] if args else f\"session-{datetime.now().strftime('%m%d-%H%M')}\"\n            return CommandResult(\n                success=True,\n                message=f\"🆕 Starting new session: {session_name}\",\n                action=\"new_session\",\n                data={\"name\": session_name}\n            )\n        \n        return CommandResult(\n            success=False,\n            message=f\"Session command not implemented: {command.name}\"\n        )\n    \n    def execute_help_command(self, command: Command, args: List[str]) -> CommandResult:\n        \"\"\"Execute help commands.\"\"\"\n        if command.name in ['help', '?', 'commands']:\n            if args:\n                # Help for specific command\n                cmd_name = args[0]\n                if cmd_name in self.commands:\n                    cmd = self.commands[cmd_name]\n                    help_text = f\"\"\"**/{cmd.name}** - {cmd.description}\n                    \n**Usage:** {cmd.usage}\n                    \n**Examples:**\n{chr(10).join(f'• {ex}' for ex in cmd.examples)}\"\"\"\n                    \n                    if cmd.parameters:\n                        help_text += f\"\\n\\n**Parameters:** {', '.join(cmd.parameters)}\"\n                    \n                    return CommandResult(\n                        success=True,\n                        message=help_text,\n                        action=\"show_help\",\n                        data={\"command\": cmd_name}\n                    )\n                else:\n                    return CommandResult(\n                        success=False,\n                        message=f\"No help available for: {cmd_name}\"\n                    )\n            else:\n                # General help\n                help_text = self.generate_general_help()\n                return CommandResult(\n                    success=True,\n                    message=help_text,\n                    action=\"show_help\"\n                )\n        \n        elif command.name in ['examples', 'sample', 'demo']:\n            category = args[0] if args else 'all'\n            examples = self.get_examples_by_category(category)\n            return CommandResult(\n                success=True,\n                message=f\"💡 Examples for {category}:\\n\\n\" + \"\\n\".join(examples),\n                action=\"show_examples\",\n                data={\"category\": category}\n            )\n        \n        return CommandResult(\n            success=False,\n            message=f\"Help command not implemented: {command.name}\"\n        )\n    \n    def find_similar_commands(self, command_name: str) -> List[str]:\n        \"\"\"Find similar commands for suggestions.\"\"\"\n        similar = []\n        \n        for cmd_name in self.commands.keys():\n            # Check if command starts with the input\n            if cmd_name.startswith(command_name[:3]):\n                similar.append(cmd_name)\n            # Check for partial matches\n            elif command_name in cmd_name or cmd_name in command_name:\n                similar.append(cmd_name)\n        \n        return similar[:3]  # Return top 3 matches\n    \n    def extract_args_from_natural_language(self, query: str, pattern: str, command_name: str) -> List[str]:\n        \"\"\"Extract arguments from natural language query.\"\"\"\n        # Simple extraction - could be enhanced with NLP\n        args = []\n        \n        if command_name == 'analyze':\n            # Extract what to analyze\n            match = re.search(r'analyze (my )?(.+)', query.lower())\n            if match:\n                args.append(match.group(2))\n        \n        elif command_name == 'report':\n            # Extract report type\n            match = re.search(r'(generate|create|make) (.+) report', query.lower())\n            if match:\n                args.append(match.group(2))\n        \n        return args\n    \n    def generate_general_help(self) -> str:\n        \"\"\"Generate general help text.\"\"\"\n        help_sections = {\n            'Navigation': ['dashboard', 'security', 'iam', 'compliance'],\n            'Analysis': ['scan', 'analyze', 'report'],\n            'Agents': ['agents', 'agent', 'transfer'],\n            'Session': ['history', 'clear', 'export', 'new'],\n            'Help': ['help', 'examples']\n        }\n        \n        help_text = \"**🤖 ADK Security Agent Commands**\\n\\n\"\n        \n        for section, commands in help_sections.items():\n            help_text += f\"**{section}:**\\n\"\n            for cmd_name in commands:\n                if cmd_name in self.commands:\n                    cmd = self.commands[cmd_name]\n                    help_text += f\"• `/{cmd_name}` - {cmd.description}\\n\"\n            help_text += \"\\n\"\n        \n        help_text += \"**💡 Tips:**\\n\"\n        help_text += \"• Use `/help <command>` for detailed help\\n\"\n        help_text += \"• Try natural language: 'show me security findings'\\n\"\n        help_text += \"• Use `/examples` to see sample queries\\n\"\n        \n        return help_text\n    \n    def get_examples_by_category(self, category: str) -> List[str]:\n        \"\"\"Get examples by category.\"\"\"\n        examples = []\n        \n        category_map = {\n            'navigation': ['dashboard', 'security', 'iam', 'compliance'],\n            'analysis': ['scan', 'analyze', 'report'],\n            'agents': ['agents', 'agent', 'transfer'],\n            'security': ['security', 'scan'],\n            'iam': ['iam'],\n            'compliance': ['compliance']\n        }\n        \n        commands_to_show = category_map.get(category.lower(), self.commands.keys())\n        \n        for cmd_name in commands_to_show:\n            if cmd_name in self.commands:\n                cmd = self.commands[cmd_name]\n                examples.extend([f\"• {ex}\" for ex in cmd.examples[:2]])  # Show first 2 examples\n        \n        return examples\n    \n    def get_command_suggestions(self, partial_input: str) -> List[str]:\n        \"\"\"Get command suggestions for auto-completion.\"\"\"\n        if not partial_input.startswith('/'):\n            return []\n        \n        partial_cmd = partial_input[1:].lower()\n        suggestions = []\n        \n        for cmd_name in self.commands.keys():\n            if cmd_name.startswith(partial_cmd):\n                suggestions.append(f\"/{cmd_name}\")\n        \n        return suggestions[:5]  # Return top 5 suggestions\n    \n    def add_to_favorites(self, command: str):\n        \"\"\"Add command to favorites.\"\"\"\n        self.favorites.add(command)\n    \n    def get_command_history(self, count: int = 10) -> List[Dict[str, Any]]:\n        \"\"\"Get recent command history.\"\"\"\n        return self.command_history[-count:] if self.command_history else []


# Global command processor instance\n_command_processor = None\n\ndef get_command_processor() -> ChatCommandProcessor:\n    \"\"\"Get or create global command processor instance.\"\"\"\n    global _command_processor\n    if _command_processor is None:\n        _command_processor = ChatCommandProcessor()\n    return _command_processor\n\ndef process_chat_input(user_input: str) -> CommandResult:\n    \"\"\"Main function to process chat input.\"\"\"\n    processor = get_command_processor()\n    return processor.process_input(user_input)