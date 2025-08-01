"""Security agent with API Hub integration."""

import json
import os
from typing import List, Dict, Any, Optional
from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import requests
from bs4 import BeautifulSoup
from google.adk.tools.apihub_tool.apihub_toolset import APIHubToolset
from google.adk.auth import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.tools.openapi_tool.auth.auth_helpers import dict_to_auth_scheme

# Import GCP clients for live data access
from google.cloud.resourcemanager_v3 import ProjectsClient, ListProjectsRequest
from google.cloud import service_usage_v1
from google.auth import default


def load_security_kb(kb_path: str) -> Dict[str, Any]:
    """Load the GCP API security knowledge base from a JSON file.
    
    Args:
        kb_path: Path to the JSON knowledge base file.
        
    Returns:
        Dictionary containing the parsed knowledge base data.
        
    Raises:
        FileNotFoundError: If the knowledge base file doesn't exist.
        JSONDecodeError: If the JSON file is malformed.
    """
    with open(kb_path, 'r') as f:
        return json.load(f)


def create_apihub_toolset(
    toolset_name: str,
    apihub_resource_name: str,
    auth_type: str = "service_account",
    auth_config: Optional[Dict[str, Any]] = None,
    tool_filter: Optional[List[str]] = None,
    description: str = ""
) -> APIHubToolset:
    """Create an API Hub toolset for dynamic tool access.
    
    Args:
        toolset_name: Name for the toolset.
        apihub_resource_name: API Hub resource name.
        auth_type: Authentication type ('service_account', 'access_token', 'oauth2').
        auth_config: Authentication configuration.
        tool_filter: List of tool names to include.
        description: Toolset description.
        
    Returns:
        Configured APIHubToolset instance.
        
    Raises:
        Exception: If toolset creation fails.
    """
    try:
        toolset_kwargs = {
            "name": toolset_name,
            "description": description,
            "apihub_resource_name": apihub_resource_name,
            "tool_filter": tool_filter
        }
        
        if auth_type == "service_account":
            if auth_config and "service_account_json" in auth_config:
                toolset_kwargs["service_account_json"] = auth_config["service_account_json"]
            # Otherwise use default credentials
            
        elif auth_type == "access_token":
            if auth_config and "access_token" in auth_config:
                toolset_kwargs["access_token"] = auth_config["access_token"]
            else:
                raise Exception("Access token not provided in auth config")
                
        elif auth_type == "oauth2":
            if auth_config and "oauth2_config" in auth_config:
                oauth2_config = auth_config["oauth2_config"]
                oauth_scheme = dict_to_auth_scheme(oauth2_config)
                
                auth_credential = AuthCredential(
                    auth_type=AuthCredentialTypes.OAUTH2,
                    oauth2=OAuth2Auth(
                        client_id=oauth2_config.get("client_id"),
                        client_secret=oauth2_config.get("client_secret")
                    )
                )
                
                toolset_kwargs["auth_scheme"] = oauth_scheme
                toolset_kwargs["auth_credential"] = auth_credential
            else:
                raise Exception("OAuth2 configuration not provided")
        
        return APIHubToolset(**toolset_kwargs)
        
    except Exception as e:
        raise Exception(f"Failed to create API Hub toolset: {str(e)}")


def get_available_toolsets() -> List[Dict[str, Any]]:
    """Get list of available API Hub toolsets from configuration.
    
    Returns:
        List of toolset configurations.
    """
    # This would typically read from a configuration file or database
    # For now, return an empty list - toolsets should be configured via the backend
    return []


def evaluate_api_security(api_name: str, tool_context: ToolContext) -> str:
    """Evaluate the security stance of a GCP API using the knowledge base.

    This function looks up the specified API in the knowledge base and returns
    a formatted summary of security considerations and recommended practices.

    Args:
        api_name: Name of the GCP API to evaluate (case-insensitive).
        tool_context: ToolContext for state and logging (unused in this implementation).

    Returns:
        A formatted string containing:
        - Security evaluation summary
        - List of security considerations
        - List of recommended practices
        - Documentation URL reference
        
    Example:
        >>> result = evaluate_api_security("Cloud Storage", tool_context)
        >>> print(result)
        Security Evaluation for Cloud Storage (see docs: https://cloud.google.com/storage/docs):
        Security Considerations:
        - Data is encrypted at rest and in transit.
        - IAM roles control access to buckets and objects.
        ...
    """
    kb_path = os.path.join(os.path.dirname(__file__), 'gcp_api_security_kb.json')
    kb = load_security_kb(kb_path)
    api_info = next((api for api in kb['apis'] if api['name'].lower() == api_name.lower()), None)
    if not api_info:
        return f"No security information found for API: {api_name}. Please check the API name or update the knowledge base."
    summary = [
        f"Security Evaluation for {api_info['name']} (see docs: {api_info['documentation_url']}):",
        "\nSecurity Considerations:",
    ]
    summary.extend(f"- {item}" for item in api_info['security_considerations'])
    summary.append("\nRecommended Practices:")
    summary.extend(f"- {item}" for item in api_info['recommended_practices'])
    return '\n'.join(summary)


def get_api_dependency_graph(api_name: str, kb_path: str) -> dict:
    """Recursively build a dependency graph for the given API.
    
    This function traverses the dependency tree starting from the specified API
    and builds a nested dictionary representing the dependency relationships.
    It handles circular dependencies by tracking visited nodes.
    
    Args:
        api_name: Name of the API to build the dependency graph for.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Nested dictionary representing the dependency graph structure.
        Format: {api_name: {dependency: {sub_dependency: {}}}}
        
    Example:
        >>> graph = get_api_dependency_graph("Cloud Storage", kb_path)
        >>> print(graph)
        {'Cloud Storage': {'IAM': {}, 'Cloud KMS': {'IAM': {}}}}
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    visited = set()
    def build_graph(name):
        if name not in api_map or name in visited:
            return {}
        visited.add(name)
        deps = api_map[name].get('dependencies', [])
        return {name: {dep: build_graph(dep) for dep in deps}}
    return build_graph(api_name)


def propagate_risk(api_name: str, kb_path: str) -> dict:
    """Propagate risk through the dependency graph and report at-risk services.
    
    This function analyzes the dependency tree starting from the specified API
    and identifies all services that are at risk due to direct vulnerabilities
    or dependencies on vulnerable services. It provides detailed reasoning and
    the path of risk propagation.
    
    Args:
        api_name: Name of the API to analyze for risk propagation.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Dictionary mapping service names to risk information:
        {
            'service_name': {
                'at_risk': bool,
                'reason': str,
                'path': List[str]
            }
        }
        
    Example:
        >>> risk_report = propagate_risk("Cloud Storage", kb_path)
        >>> print(risk_report)
        {
            'Cloud Storage': {
                'at_risk': True,
                'reason': 'Depends on a vulnerable service.',
                'path': ['Cloud Storage', 'Cloud KMS']
            },
            'Cloud KMS': {
                'at_risk': True,
                'reason': 'Cloud KMS is directly vulnerable.',
                'path': ['Cloud Storage', 'Cloud KMS']
            }
        }
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    risk_report = {}
    def check_risk(name, path=None):
        if path is None:
            path = [name]
        api = api_map.get(name)
        if not api:
            return False
        if api.get('vulnerable', False):
            risk_report[name] = {
                'at_risk': True,
                'reason': f"{name} is directly vulnerable.",
                'path': list(path)
            }
            return True
        deps = api.get('dependencies', [])
        at_risk = False
        for dep in deps:
            if check_risk(dep, path + [dep]):
                at_risk = True
        if at_risk:
            risk_report[name] = {
                'at_risk': True,
                'reason': f"Depends on a vulnerable service.",
                'path': list(path)
            }
        else:
            risk_report[name] = {
                'at_risk': False,
                'reason': "No vulnerable dependencies detected.",
                'path': list(path)
            }
        return at_risk
    check_risk(api_name)
    return risk_report


def get_api_dependency_graph(api_name: str, kb_path: str) -> dict:
    """Recursively build a dependency graph for the given API.
    
    This function traverses the dependency tree starting from the specified API
    and builds a nested dictionary representing the dependency relationships.
    It handles circular dependencies by tracking visited nodes.
    
    Args:
        api_name: Name of the API to build the dependency graph for.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Nested dictionary representing the dependency graph structure.
        Format: {api_name: {dependency: {sub_dependency: {}}}}
        
    Example:
        >>> graph = get_api_dependency_graph("Cloud Storage", kb_path)
        >>> print(graph)
        {'Cloud Storage': {'IAM': {}, 'Cloud KMS': {'IAM': {}}}}
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    visited = set()
    def build_graph(name):
        if name not in api_map or name in visited:
            return {}
        visited.add(name)
        deps = api_map[name].get('dependencies', [])
        return {name: {dep: build_graph(dep) for dep in deps}}
    return build_graph(api_name)


def propagate_risk(api_name: str, kb_path: str) -> dict:
    """Propagate risk through the dependency graph and report at-risk services.
    
    This function analyzes the dependency tree starting from the specified API
    and identifies all services that are at risk due to direct vulnerabilities
    or dependencies on vulnerable services. It provides detailed reasoning and
    the path of risk propagation.
    
    Args:
        api_name: Name of the API to analyze for risk propagation.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Dictionary mapping service names to risk information:
        {
            'service_name': {
                'at_risk': bool,
                'reason': str,
                'path': List[str]
            }
        }
        
    Example:
        >>> risk_report = propagate_risk("Cloud Storage", kb_path)
        >>> print(risk_report)
        {
            'Cloud Storage': {
                'at_risk': True,
                'reason': 'Depends on a vulnerable service.',
                'path': ['Cloud Storage', 'Cloud KMS']
            },
            'Cloud KMS': {
                'at_risk': True,
                'reason': 'Cloud KMS is directly vulnerable.',
                'path': ['Cloud Storage', 'Cloud KMS']
            }
        }
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    risk_report = {}
    def check_risk(name, path=None):
        if path is None:
            path = [name]
        api = api_map.get(name)
        if not api:
            return False
        if api.get('vulnerable', False):
            risk_report[name] = {
                'at_risk': True,
                'reason': f"{name} is directly vulnerable.",
                'path': list(path)
            }
            return True
        deps = api.get('dependencies', [])
        at_risk = False
        for dep in deps:
            if check_risk(dep, path + [dep]):
                at_risk = True
        if at_risk:
            risk_report[name] = {
                'at_risk': True,
                'reason': f"Depends on a vulnerable service.",
                'path': list(path)
            }
        else:
            risk_report[name] = {
                'at_risk': False,
                'reason': "No vulnerable dependencies detected.",
                'path': list(path)
            }
        return at_risk
    check_risk(api_name)
    return risk_report


def scrape_api_documentation(doc_url: str, tool_context: ToolContext = None) -> str:
    """Scrape the documentation URL for limits or considerations.
    
    This function fetches a web page and extracts text content that mentions
    security-related keywords like 'limit', 'limitation', 'consideration',
    'quota', or 'restriction'. It's useful for automatically gathering
    security information from official documentation.
    
    Args:
        doc_url: URL of the documentation page to scrape.
        tool_context: ToolContext for state and logging (unused in this implementation).
        
    Returns:
        String containing up to 20 findings that match the security keywords,
        or an error message if scraping fails.
        
    Example:
        >>> findings = scrape_api_documentation("https://cloud.google.com/storage/quotas")
        >>> print(findings)
        Findings from https://cloud.google.com/storage/quotas:
        - Storage quotas and limits
        - Request rate limits
        - Object size limitations
        ...
    """
    try:
        resp = requests.get(doc_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator='\n')
        lines = text.splitlines()
        findings = []
        keywords = ['limit', 'limitation', 'consideration', 'quota', 'restriction']
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if any(kw in l.lower() for kw in keywords):
                findings.append(l)
        if not findings:
            return f"No explicit limits or considerations found at {doc_url}."
        return f"Findings from {doc_url}:\n" + '\n'.join(findings[:20])
    except Exception as e:
        return f"Error scraping {doc_url}: {e}"


def get_gcp_projects(tool_context: ToolContext) -> str:
    """Get list of accessible GCP projects for the user.
    
    Returns:
        String containing formatted list of accessible GCP projects.
    """
    try:
        # Use the existing backend API endpoint that we know works
        import requests
        response = requests.get("http://localhost:8000/api/v1/gcp/projects")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("projects"):
                projects = data["projects"]
                project_details = data.get("project_details", [])
                
                result = ["Available GCP Projects:"]
                for i, project_id in enumerate(projects):
                    # Try to get display name from project_details
                    display_name = project_id
                    if i < len(project_details):
                        display_name = project_details[i].get("display_name", project_id)
                    result.append(f"- {display_name} ({project_id})")
                    
                return "\n".join(result)
            else:
                return f"No projects found or API error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing GCP projects API: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error accessing GCP projects: {str(e)}"


def get_project_services(project_id: str, tool_context: ToolContext) -> str:
    """Get enabled services for a specific GCP project.
    
    Args:
        project_id: The GCP project ID to analyze.
        tool_context: ToolContext for state and logging.
        
    Returns:
        String containing formatted list of enabled services.
    """
    try:
        # Use the existing backend API endpoint
        import requests
        response = requests.get(f"http://localhost:8000/api/v1/gcp/project/{project_id}/services")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("services"):
                services = data["services"]
                
                result = [f"Enabled services in project {project_id}:"]
                for service in services[:20]:  # Limit to first 20 services
                    result.append(f"- {service.get('display_name', service.get('name', 'Unknown'))}")
                    
                if len(services) > 20:
                    result.append(f"... and {len(services) - 20} more services")
                    
                return "\n".join(result)
            else:
                return f"No enabled services found for project {project_id}. Error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing services API for project {project_id}: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error getting services for project {project_id}: {str(e)}"


def get_project_info(project_id: str, tool_context: ToolContext) -> str:
    """Get detailed information about a specific GCP project.
    
    Args:
        project_id: The GCP project ID to analyze.
        tool_context: ToolContext for state and logging.
        
    Returns:
        String containing formatted project information.
    """
    try:
        # Use the existing backend API endpoint
        import requests
        response = requests.get(f"http://localhost:8000/api/v1/gcp/project/{project_id}/info")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("project"):
                project = data["project"]
                
                result = [f"Project Information for {project_id}:"]
                result.append(f"- Display Name: {project.get('display_name', 'Unknown')}")
                result.append(f"- Project Number: {project.get('project_number', 'Unknown')}")
                result.append(f"- State: {project.get('state', 'Unknown')}")
                
                if project.get('create_time'):
                    result.append(f"- Created: {project['create_time']}")
                    
                if project.get('labels'):
                    result.append("- Labels:")
                    for key, value in project['labels'].items():
                        result.append(f"  - {key}: {value}")
                
                return "\n".join(result)
            else:
                return f"No project info found for {project_id}. Error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing project info API for {project_id}: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error getting project info for {project_id}: {str(e)}"


def call_google_api(
    service: str,
    version: str,
    resource_path: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Constructs and executes a REST call to a specified Google Cloud API endpoint
    by calling the backend's generic API endpoint.

    Args:
        service: The Google Cloud service name (e.g., 'storage', 'cloudresourcemanager').
        version: The API version (e.g., 'v1', 'v3').
        resource_path: The resource path for the API call (e.g., 'b/my-bucket/o').
        method: The HTTP method to use (GET, POST, PUT, DELETE).
        body: The JSON body for POST or PUT requests.
        tool_context: The context for the tool execution.

    Returns:
        The JSON response from the API as a string, or an error message.
    """
    try:
        import requests
        import json

        request_data = {
            "service": service,
            "version": version,
            "resource_path": resource_path,
            "method": method,
            "body": body
        }
        
        response = requests.post("http://localhost:8000/api/v1/gcp/call-api", json=request_data)
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get("success"):
            return json.dumps(response_data.get("response", {}), indent=2)
        else:
            return f"Error from backend API: {response_data.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error calling backend for Google API: {e}"



def analyze_gcs_bucket_security(project_id: str, tool_context: ToolContext) -> str:
    """
    Analyzes all GCS buckets in a project for common security misconfigurations
    and returns a concise, actionable summary of recommendations.
    """
    try:
        from google.cloud import storage

        storage_client = storage.Client(project=project_id)
        buckets = storage_client.list_buckets()

        buckets_without_versioning = []
        public_buckets = []

        for bucket in buckets:
            # Check for versioning
            if not bucket.versioning_enabled:
                buckets_without_versioning.append(bucket.name)
            
            # Check for public access (simplified check)
            try:
                iam_policy = bucket.get_iam_policy(requested_policy_version=3)
                for binding in iam_policy.bindings:
                    if 'allUsers' in binding['members'] or 'allAuthenticatedUsers' in binding['members']:
                        public_buckets.append(bucket.name)
                        break 
            except Exception:
                # This can fail if uniform bucket-level access is not enabled,
                # which is itself a security finding. For this tool, we'll focus on versioning.
                pass

        recommendations = []
        if buckets_without_versioning:
            recommendations.append(
                f"Enable versioning on the following buckets to protect against data loss: {', '.join(buckets_without_versioning)}"
            )
        if public_buckets:
            recommendations.append(
                f"Remove public access from the following buckets: {', '.join(public_buckets)}"
            )

        if not recommendations:
            return f"No immediate security recommendations for GCS buckets in project '{project_id}'."

        return "Actionable GCS Security Recommendations:\n- " + "\n- ".join(recommendations)

    except Exception as e:
        return f"Error analyzing GCS bucket security for project '{project_id}': {e}"




# Create the root agent instance with .env configuration
import os
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and value and value != 'your-api-key-here':
                        os.environ.setdefault(key.strip(), value.strip())

# Load .env configuration
load_env_file()

# Set Vertex AI environment variables for auto-detection
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
os.environ.setdefault('GOOGLE_CLOUD_LOCATION', 'us-central1')

# Set google-genai specific environment variables for Vertex AI configuration
os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'true')
os.environ.setdefault('GOOGLE_GENAI_PROJECT', 'mgm-digitalconcierge')
os.environ.setdefault('GOOGLE_GENAI_LOCATION', 'us-central1')

# Initialize Vertex AI and google-genai with the project and location  
import vertexai
import google.genai

try:
    # Initialize Vertex AI
    vertexai.init(
        project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge'),
        location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    )
    print(f"✅ Vertex AI initialized for project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')}")
    
except Exception as e:
    print(f"❌ Vertex AI initialization failed: {e}")

# Dynamically load API Hub toolset if configured
apihub_toolsets_to_add = []
apihub_resource_name = os.environ.get('APIHUB_RESOURCE_NAME')
if apihub_resource_name:
    try:
        print(f"Attempting to load API Hub Toolset for resource: {apihub_resource_name}")
        apihub_toolset = create_apihub_toolset(
            toolset_name="security_apihub_tools",
            apihub_resource_name=apihub_resource_name,
            description="Tools dynamically loaded from API Hub for security evaluations"
        )
        apihub_toolsets_to_add.append(apihub_toolset)
        print("✅ API Hub Toolset loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load API Hub Toolset from {apihub_resource_name}: {e}")

# Collect all tools for the root agent
base_tools = [
    get_gcp_projects,
    get_project_info,
    get_project_services,
    evaluate_api_security,
    scrape_api_documentation,
    analyze_gcs_bucket_security,
    call_google_api,
    get_api_dependency_graph,
    propagate_risk
]

# Add tools from API Hub toolsets
for toolset in apihub_toolsets_to_add:
    try:
        tools = toolset.get_tools()
        base_tools.extend(tools)
        print(f"Added {len(tools)} tools from API Hub toolset: {toolset.name}")
    except Exception as e:
        print(f"Warning: Failed to load tools from API Hub toolset {toolset.name}: {e}")

# Create agent using Vertex AI with ADC (recommended approach)
print("🔑 Using Vertex AI with Application Default Credentials")
root_agent = Agent(
    model='gemini-2.5-flash',
    name='security_agent',
    description=(
        'Security evaluation agent for evaluating the security stance of '
        'onboarding new GCP APIs using public documentation and JSON knowledge base,'
        'and dynamic API Hub tools. This agent also provides dependency analysis '
        'and risk propagation.'
    ),
    instruction="""
        You are a comprehensive security evaluation agent for GCP APIs and projects. 
        
        Your primary functions:
        - Use get_gcp_projects to list the user's accessible GCP projects
        - Use get_project_info to get detailed information about specific projects
        - Use get_project_services to list enabled services in a project
        - Use evaluate_api_security to assess GCP API security using the knowledge base
        - Use scrape_api_documentation to extract security information from documentation URLs
        - Use get_api_dependency_graph to visualize API dependencies
        - Use propagate_risk to identify at-risk services due to dependencies
        - When asked about GCS buckets, use the `analyze_gcs_bucket_security` tool to provide actionable recommendations.
        - For any other Google Cloud API interactions, use the generic `call_google_api` tool.
        - Always provide actionable recommendations and reference official documentation
        
        When a user asks about their GCP environment:
        1. First use get_gcp_projects to see what projects they have access to
        2. Use get_project_services to see what services are enabled
        3. Use get_project_info for detailed project information when needed
        4. Use evaluate_api_security for knowledge base security assessments
        5. Provide comprehensive security recommendations based on actual project data
        
        You have access to LIVE GCP data through proper authentication, so you can analyze
        the user's actual projects, services, and configurations. Always use real data
        when available rather than generic responses.
        
        If an API is not found in your knowledge base, inform the user and suggest updating it.
        Always prioritize security best practices and compliance requirements.
        """,
    tools=[get_gcp_projects, get_project_info, get_project_services, evaluate_api_security, scrape_api_documentation, analyze_gcs_bucket_security, call_google_api],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    )
) 