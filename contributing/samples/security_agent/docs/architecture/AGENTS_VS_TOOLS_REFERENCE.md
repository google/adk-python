# 🤖 Agents vs 🔧 Tools: Quick Reference

## TL;DR
- **🤖 AGENTS** = The "brain" - intelligent reasoning, conversation, coordination
- **🔧 TOOLS** = The "hands" - specific functions, API calls, data processing

## 🤖 AGENTS (Intelligent Coordinators)

### What They Do
- **Think**: Use LLMs to understand and reason about user requests
- **Decide**: Choose which tools to use and how to use them
- **Talk**: Convert between human language and technical operations
- **Remember**: Maintain conversation context and learn from interactions
- **Coordinate**: Orchestrate complex workflows across multiple tools

### Current Agents
| Agent | Purpose | Uses Tools |
|-------|---------|------------|
| **CoordinatorAgent** | Main orchestrator, query routing | All tools via delegation |
| **StorageSecurityAgent** | GCS bucket security analysis | `storage_tools.py`, `gcp_tools` |
| **IAMSecurityAgent** | Identity & access management | IAM APIs, security analysis |
| **NetworkSecurityAgent** | Firewall and network security | Network APIs, compliance checks |
| **ComplianceAgent** | SOC2, GDPR, ISO compliance | Knowledge base, audit tools |

### Example Agent Interaction
```
User: "Are my storage buckets secure?"

CoordinatorAgent thinks:
- This is about storage security
- Route to StorageSecurityAgent
- Need bucket analysis and recommendations

StorageSecurityAgent:
- Use storage_tools to scan buckets
- Analyze findings with security knowledge
- Generate human-readable recommendations
- "Found 3 buckets, 1 has public access. Here's how to fix it..."
```

## 🔧 TOOLS (Specialized Functions)

### What They Do
- **Execute**: Perform specific, focused tasks
- **Connect**: Interface with external APIs and services  
- **Process**: Transform and analyze data
- **Return**: Provide structured, predictable results
- **Operate**: Work without state or memory

### Current Tools
| Tool Category | Tools | Purpose |
|---------------|-------|---------|
| **GCP Tools** | `storage_tools.py`, `project_tools.py` | Google Cloud API interactions |
| **Security Tools** | `knowledge_base_tools.py` | Security best practices and analysis |
| **Analysis Tools** | `dependency_analysis.py` | Data processing and reporting |
| **API Tools** | `google_api_tools.py` | Generic API interactions |

### Example Tool Function
```python
def analyze_gcs_bucket_security(project_id: str, tool_context: ToolContext) -> str:
    """
    INPUT: project_id = "my-gcp-project"
    
    PROCESS:
    1. Call GCS API to list buckets
    2. Check each bucket's IAM policies
    3. Identify public access permissions
    4. Check versioning settings
    
    OUTPUT: JSON with findings and recommendations
    """
```

## 🔄 How They Work Together

### The Flow
1. **User** speaks to **Agent** in natural language
2. **Agent** understands intent and selects appropriate **Tools**
3. **Tools** execute specific tasks and return structured data
4. **Agent** synthesizes results and responds in natural language
5. **Agent** maintains context for follow-up questions

### Example Workflow
```
User → "Check my project's security"

CoordinatorAgent:
  ├─ Calls storage_tools.analyze_gcs_bucket_security()
  ├─ Calls iam_tools.analyze_project_permissions() 
  ├─ Calls network_tools.analyze_firewall_rules()
  └─ Synthesizes: "Found 3 security issues. Here's how to fix them..."

User → "How do I fix the storage issue?"

CoordinatorAgent:
  ├─ Remembers previous analysis context
  ├─ Calls storage_tools.get_remediation_commands()
  └─ Provides: "Run these commands: gsutil iam ch -d allUsers..."
```

## ⚡ Key Principles

### **Agents Should**
✅ Use natural language  
✅ Make intelligent decisions  
✅ Maintain conversation context  
✅ Coordinate multiple tools  
✅ Provide explanations and reasoning  

### **Agents Should NOT**
❌ Make direct API calls  
❌ Perform low-level data processing  
❌ Be stateless  
❌ Return raw technical data  

### **Tools Should**  
✅ Have single, clear purpose  
✅ Return predictable, structured output  
✅ Be stateless and deterministic  
✅ Handle errors gracefully  
✅ Focus on specific functionality  

### **Tools Should NOT**
❌ Make decisions about what to do next  
❌ Interact directly with users  
❌ Maintain conversation state  
❌ Generate natural language explanations  
❌ Coordinate with other tools  

## 🎯 When to Create New Agents vs Tools

### Create a New **Agent** When:
- Users need to have conversations about a new domain
- Complex multi-step reasoning is required
- Multiple tools need coordination
- Context and memory are important
- Example: "DataPrivacyAgent" for GDPR compliance conversations

### Create a New **Tool** When:
- A specific API or service needs integration
- A particular calculation or analysis is needed
- A focused task can be clearly defined
- The function will be reused by multiple agents
- Example: `analyze_cloud_sql_security()` for database security checks

## 🚀 Current Architecture Summary

```
Frontend (Streamlit)
    ↓ HTTP
Backend API (FastAPI)
    ↓ Session Management
AGENTS Layer
    ├─ CoordinatorAgent (main orchestrator)
    ├─ StorageSecurityAgent (bucket specialist)
    ├─ IAMSecurityAgent (permissions specialist)
    └─ [More specialists...]
    ↓ Tool Calls
TOOLS Layer
    ├─ GCP Tools (storage, project, IAM)
    ├─ Security Tools (knowledge, scanning)
    └─ Analysis Tools (reporting, dependencies)
    ↓ API Calls
External Services (Google Cloud, APIs)
```

This architecture provides **intelligence** (via Agents) and **capability** (via Tools) while maintaining clear separation of concerns and ADK best practices.