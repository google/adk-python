"""
Asset Discovery Agent - Intelligent routing for any GCP service query using Asset Inventory API
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AssetDiscoveryAgent:
    """
    Intelligent agent that can handle queries about ANY GCP service using Asset Inventory API
    Routes natural language queries to appropriate asset discovery endpoints
    """
    
    def __init__(self, project_id: str, base_url: str = "http://localhost:8000"):
        self.project_id = project_id
        self.base_url = base_url
        self.agent_type = "asset_discovery"
        self.description = f"Asset Discovery specialist for comprehensive GCP resource analysis in project {project_id}"
        
    def can_handle_query(self, query: str) -> bool:
        """Check if this agent can handle the query based on GCP service keywords"""
        gcp_keywords = [
            # Compute
            'compute', 'instance', 'vm', 'virtual machine', 'server', 'disk',
            # Storage  
            'storage', 'bucket', 'blob', 'disk', 'snapshot',
            # Databases
            'database', 'sql', 'mysql', 'postgres', 'spanner', 'bigquery', 'datastore',
            # Networking
            'network', 'vpc', 'firewall', 'subnet', 'load balancer', 'dns',
            # Containers
            'kubernetes', 'gke', 'cluster', 'container', 'pod',
            # Serverless
            'function', 'cloud function', 'app engine', 'cloud run',
            # AI/ML
            'vertex', 'ai', 'ml', 'model', 'training',
            # Security
            'iam', 'security', 'policy', 'role', 'permission', 'secret',
            # Monitoring
            'monitoring', 'logging', 'alert', 'metric',
            # General discovery
            'resources', 'project', 'inventory', 'assets', 'services'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in gcp_keywords)
    
    async def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process natural language queries about any GCP service using Asset Inventory
        """
        logger.info(f"🔍 [ASSET-{session_id}] Processing asset discovery query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Route query to appropriate asset discovery endpoint
            if self._is_summary_query(query):
                response = await self._get_project_summary()
            elif self._is_compute_query(query):
                response = await self._get_compute_resources()
            elif self._is_storage_query(query):
                response = await self._get_storage_resources()
            elif self._is_database_query(query):
                response = await self._get_database_resources()
            elif self._is_network_query(query):
                response = await self._get_network_resources()
            else:
                # General discovery using natural language endpoint
                response = await self._discover_resources(query)
            
            processing_time = time.time() - start_time
            
            # Format response for chat interface
            formatted_response = self._format_response(response, query, processing_time)
            
            logger.info(f"✅ [ASSET-{session_id}] Asset discovery completed in {processing_time:.2f}s")
            
            return {
                "success": True,
                "response": formatted_response,
                "agent_used": "AssetDiscoveryAgent",
                "query_type": self._get_query_type(query),
                "data_source": response.get("data_source", "unknown"),
                "api_duration": response.get("api_duration", 0),
                "resource_count": response.get("total_count", 0),
                "suggestions": self._generate_suggestions(query, response)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ [ASSET-{session_id}] Asset discovery failed: {e}")
            
            return {
                "success": False,
                "response": f"I encountered an error while discovering GCP resources: {str(e)}",
                "agent_used": "AssetDiscoveryAgent",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _is_summary_query(self, query: str) -> bool:
        """Check if query is asking for a project summary"""
        summary_terms = ['overview', 'summary', 'all resources', 'everything', 'project resources']
        return any(term in query.lower() for term in summary_terms)
    
    def _is_compute_query(self, query: str) -> bool:
        """Check if query is specifically about compute resources"""
        compute_terms = ['compute', 'instance', 'vm', 'virtual machine', 'server']
        return any(term in query.lower() for term in compute_terms)
    
    def _is_storage_query(self, query: str) -> bool:
        """Check if query is specifically about storage resources"""
        storage_terms = ['storage', 'bucket', 'blob', 'disk']
        return any(term in query.lower() for term in storage_terms)
    
    def _is_database_query(self, query: str) -> bool:
        """Check if query is specifically about database resources"""
        database_terms = ['database', 'sql', 'mysql', 'postgres', 'spanner', 'bigquery']
        return any(term in query.lower() for term in database_terms)
    
    def _is_network_query(self, query: str) -> bool:
        """Check if query is specifically about network resources"""
        network_terms = ['network', 'vpc', 'firewall', 'subnet', 'load balancer']
        return any(term in query.lower() for term in network_terms)
    
    async def _discover_resources(self, query: str) -> Dict[str, Any]:
        """Use natural language discovery endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/assets/discover",
                json={"query": query, "project_id": self.project_id},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_project_summary(self) -> Dict[str, Any]:
        """Get comprehensive project summary"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/assets/summary",
                params={"project_id": self.project_id},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_compute_resources(self) -> Dict[str, Any]:
        """Get compute instances"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/assets/compute/instances",
                params={"project_id": self.project_id},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_storage_resources(self) -> Dict[str, Any]:
        """Get storage resources"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/assets/storage/all",
                params={"project_id": self.project_id},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_database_resources(self) -> Dict[str, Any]:
        """Get database resources"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/assets/databases",
                params={"project_id": self.project_id},
                timeout=30
            )
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_network_resources(self) -> Dict[str, Any]:
        """Get network resources using general discovery with network focus"""
        return await self._discover_resources("show me network resources vpc firewall")
    
    def _format_response(self, api_response: Dict[str, Any], query: str, processing_time: float) -> str:
        """Format API response for chat interface"""
        
        if "error" in api_response:
            return f"❌ **Asset Discovery Error**\\n\\nI couldn't retrieve GCP resources: {api_response['error']}"
        
        # Handle different response types
        if "summary" in api_response:
            return self._format_summary_response(api_response)
        elif "instances" in api_response:
            return self._format_compute_response(api_response)
        elif "storage_resources" in api_response:
            return self._format_storage_response(api_response)
        elif "databases" in api_response:
            return self._format_database_response(api_response)
        elif "resources" in api_response:
            return self._format_discovery_response(api_response, query)
        else:
            return f"📊 **GCP Resource Discovery**\\n\\nFound resources in project {self.project_id}\\n\\nProcessing time: {processing_time:.2f}s"
    
    def _format_summary_response(self, response: Dict[str, Any]) -> str:
        """Format project summary response"""
        summary = response.get("summary", {})
        data_source = "🌐 Real-time data" if response.get("data_source") == "real_api" else "📋 Sample data"
        
        return f"""🏗️ **Project Resource Summary: {self.project_id}**

{data_source} via Google Cloud Asset Inventory API

**📊 Resource Overview:**
• **Compute Instances:** {summary.get('compute_instances', 0)}
• **Storage Buckets:** {summary.get('storage_buckets', 0)}
• **Databases:** {summary.get('databases', 0)}
• **Kubernetes Clusters:** {summary.get('kubernetes_clusters', 0)}
• **Cloud Functions:** {summary.get('cloud_functions', 0)}

**📈 Total Resources:** {response.get('total_resources', 0)}
**⏱️ API Response Time:** {response.get('api_duration', 0):.2f}s
**📂 Categories Found:** {', '.join(response.get('categories', []))}"""
    
    def _format_compute_response(self, response: Dict[str, Any]) -> str:
        """Format compute instances response"""
        instances = response.get("instances", [])
        data_source = "🌐 Real-time data" if response.get("data_source") == "real_api" else "📋 Sample data"
        
        if not instances:
            return f"💻 **Compute Instances: {self.project_id}**\\n\\n{data_source}\\n\\n**No compute instances found**"
        
        instance_list = []
        for instance in instances[:5]:  # Show first 5
            name = instance.get('name', 'Unknown')
            location = instance.get('location', 'Unknown')
            status = instance.get('status', instance.get('state', 'Unknown'))
            instance_list.append(f"• **{name}** - {location} ({status})")
        
        return f"""💻 **Compute Instances: {self.project_id}**

{data_source} via Google Cloud Asset Inventory API

**🖥️ Instances Found ({len(instances)}):**
{chr(10).join(instance_list)}

**⏱️ API Response Time:** {response.get('api_duration', 0):.2f}s"""
    
    def _format_storage_response(self, response: Dict[str, Any]) -> str:
        """Format storage resources response"""
        storage_resources = response.get("storage_resources", [])
        data_source = "🌐 Real-time data" if response.get("data_source") == "real_api" else "📋 Sample data"
        
        if not storage_resources:
            return f"🗄️ **Storage Resources: {self.project_id}**\\n\\n{data_source}\\n\\n**No storage resources found**"
        
        resource_list = []
        for resource in storage_resources[:5]:  # Show first 5
            name = resource.get('name', 'Unknown')
            asset_type = resource.get('asset_type', '').split('/')[-1]  # Get just the type name
            location = resource.get('location', 'Unknown')
            resource_list.append(f"• **{name}** ({asset_type}) - {location}")
        
        return f"""🗄️ **Storage Resources: {self.project_id}**

{data_source} via Google Cloud Asset Inventory API

**📦 Resources Found ({len(storage_resources)}):**
{chr(10).join(resource_list)}

**⏱️ API Response Time:** {response.get('api_duration', 0):.2f}s"""
    
    def _format_database_response(self, response: Dict[str, Any]) -> str:
        """Format database resources response"""
        databases = response.get("databases", [])
        data_source = "🌐 Real-time data" if response.get("data_source") == "real_api" else "📋 Sample data"
        
        if not databases:
            return f"🗃️ **Database Resources: {self.project_id}**\\n\\n{data_source}\\n\\n**No database resources found**"
        
        db_list = []
        for db in databases[:5]:  # Show first 5
            name = db.get('name', 'Unknown')
            db_type = db.get('asset_type', '').split('/')[-1]  # Get just the type name
            location = db.get('location', 'Unknown')
            db_list.append(f"• **{name}** ({db_type}) - {location}")
        
        return f"""🗃️ **Database Resources: {self.project_id}**

{data_source} via Google Cloud Asset Inventory API

**💾 Databases Found ({len(databases)}):**
{chr(10).join(db_list)}

**⏱️ API Response Time:** {response.get('api_duration', 0):.2f}s"""
    
    def _format_discovery_response(self, response: Dict[str, Any], query: str) -> str:
        """Format general discovery response"""
        resources = response.get("resources", {})
        data_source = "🌐 Real-time data" if response.get("data_source") == "real_api" else "📋 Sample data"
        
        if not resources:
            return f"🔍 **Resource Discovery: '{query}'**\\n\\n{data_source}\\n\\n**No resources found matching your query**"
        
        category_summary = []
        for category, resource_list in resources.items():
            if resource_list:
                category_summary.append(f"• **{category.title()}:** {len(resource_list)} resources")
        
        return f"""🔍 **Resource Discovery: "{query}"**

{data_source} via Google Cloud Asset Inventory API

**📋 Resources Found ({response.get('total_count', 0)}):**
{chr(10).join(category_summary) if category_summary else "No resources found"}

**⏱️ API Response Time:** {response.get('api_duration', 0):.2f}s
**🔍 Asset Types Searched:** {len(response.get('asset_types_searched', []))}"""
    
    def _get_query_type(self, query: str) -> str:
        """Determine the type of query for analytics"""
        query_lower = query.lower()
        if any(term in query_lower for term in ['summary', 'overview', 'all']):
            return 'summary'
        elif any(term in query_lower for term in ['compute', 'instance', 'vm']):
            return 'compute'
        elif any(term in query_lower for term in ['storage', 'bucket']):
            return 'storage'
        elif any(term in query_lower for term in ['database', 'sql']):
            return 'database'
        elif any(term in query_lower for term in ['network', 'vpc']):
            return 'network'
        else:
            return 'discovery'
    
    def _generate_suggestions(self, query: str, response: Dict[str, Any]) -> List[str]:
        """Generate intelligent follow-up suggestions based on query and response"""
        suggestions = []
        
        # Base suggestions for discovered resources
        if response.get("total_count", 0) > 0:
            suggestions.extend([
                "Show me a project summary",
                "Analyze security risks for these resources",
                "What are the cost implications?"
            ])
        
        # Query-specific suggestions
        query_lower = query.lower()
        if 'compute' in query_lower:
            suggestions.extend([
                "Show me storage resources",
                "What databases are running?",
                "Check network configuration"
            ])
        elif 'storage' in query_lower:
            suggestions.extend([
                "Show me compute instances",
                "What are my database resources?",
                "Analyze storage costs"
            ])
        elif 'database' in query_lower:
            suggestions.extend([
                "Show me compute resources",
                "What storage is being used?",
                "Check database security"
            ])
        else:
            suggestions.extend([
                "Show me compute instances",
                "What storage resources exist?",
                "List all databases"
            ])
        
        return suggestions[:5]  # Return max 5 suggestions


def create_asset_discovery_agent(project_id: str) -> AssetDiscoveryAgent:
    """Factory function to create an Asset Discovery Agent"""
    return AssetDiscoveryAgent(project_id)