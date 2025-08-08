"""Real GCP Compute API Service using existing patterns."""

import logging
from typing import Dict, Any, List
import asyncio
from ..gcp.service import GCPService

logger = logging.getLogger(__name__)

class ComputeAPIService:
    """Service to get real compute resources using GCP REST APIs."""
    
    def __init__(self, credentials=None, project_id=None):
        self.project_id = project_id
        self.gcp_service = GCPService(credentials=credentials, project_id=project_id)
        
    async def get_compute_inventory(self) -> Dict[str, Any]:
        """Get compute inventory using REST API calls."""
        try:
            logger.info(f"🔍 Fetching compute inventory for project: {self.project_id}")
            
            # Run API calls concurrently
            tasks = [
                self._get_vm_instances(),
                self._get_instance_groups(),
                self._get_load_balancers(),
                self._get_app_engine_info()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vm_data, ig_data, lb_data, ae_data = results
            
            # Handle any exceptions
            if isinstance(vm_data, Exception):
                logger.error(f"VM instances error: {vm_data}")
                vm_data = {"total": 0, "running": 0, "stopped": 0, "instances": [], "error": str(vm_data)}
                
            if isinstance(ig_data, Exception):
                logger.error(f"Instance groups error: {ig_data}")
                ig_data = {"total": 0, "groups": [], "error": str(ig_data)}
                
            if isinstance(lb_data, Exception):
                logger.error(f"Load balancers error: {lb_data}")
                lb_data = {"total": 0, "regional": [], "global": [], "error": str(lb_data)}
                
            if isinstance(ae_data, Exception):
                logger.error(f"App Engine error: {ae_data}")
                ae_data = {"exists": False, "error": str(ae_data)}
            
            return {
                "vm_instances": vm_data,
                "instance_groups": ig_data, 
                "load_balancers": lb_data,
                "app_engine": ae_data
            }
            
        except Exception as e:
            logger.error(f"Error getting compute inventory: {e}")
            return self._get_fallback_data()
    
    async def _get_vm_instances(self) -> Dict[str, Any]:
        """Get VM instances using Compute API."""
        try:
            # Get list of zones first
            zones_response = self.gcp_service.call_google_api(
                service="compute",
                version="v1", 
                resource_path=f"projects/{self.project_id}/zones"
            )
            
            all_instances = []
            running_count = 0
            stopped_count = 0
            
            if "items" in zones_response:
                for zone in zones_response["items"][:5]:  # Limit to first 5 zones for performance
                    zone_name = zone["name"]
                    try:
                        # List instances in each zone
                        instances_response = self.gcp_service.call_google_api(
                            service="compute",
                            version="v1",
                            resource_path=f"projects/{self.project_id}/zones/{zone_name}/instances"
                        )
                        
                        if "items" in instances_response:
                            for instance in instances_response["items"]:
                                all_instances.append({
                                    "name": instance.get("name", "unknown"),
                                    "zone": zone_name,
                                    "status": instance.get("status", "unknown"),
                                    "machine_type": instance.get("machineType", "").split("/")[-1],
                                    "creation_timestamp": instance.get("creationTimestamp", "")
                                })
                                
                                # Count by status
                                if instance.get("status") == "RUNNING":
                                    running_count += 1
                                elif instance.get("status") in ["STOPPED", "TERMINATED"]:
                                    stopped_count += 1
                                    
                    except Exception as zone_error:
                        logger.warning(f"Error getting instances from zone {zone_name}: {zone_error}")
                        continue
            
            logger.info(f"✅ Found {len(all_instances)} VM instances ({running_count} running, {stopped_count} stopped)")
            
            return {
                "total": len(all_instances),
                "running": running_count, 
                "stopped": stopped_count,
                "instances": all_instances[:10],  # Limit for display
                "api_calls": [f"compute.instances.list in {len(zones_response.get('items', []))} zones"]
            }
            
        except Exception as e:
            logger.error(f"Error getting VM instances: {e}")
            return {"total": 0, "running": 0, "stopped": 0, "instances": [], "error": str(e)}
    
    async def _get_instance_groups(self) -> Dict[str, Any]:
        """Get managed instance groups."""
        try:
            # Get regions
            regions_response = self.gcp_service.call_google_api(
                service="compute",
                version="v1",
                resource_path=f"projects/{self.project_id}/regions"
            )
            
            all_groups = []
            
            if "items" in regions_response:
                for region in regions_response["items"][:3]:  # Limit for performance
                    region_name = region["name"]
                    try:
                        # Get managed instance groups
                        groups_response = self.gcp_service.call_google_api(
                            service="compute",
                            version="v1",
                            resource_path=f"projects/{self.project_id}/regions/{region_name}/instanceGroupManagers"
                        )
                        
                        if "items" in groups_response:
                            for group in groups_response["items"]:
                                all_groups.append({
                                    "name": group.get("name", "unknown"),
                                    "region": region_name,
                                    "target_size": group.get("targetSize", 0),
                                    "creation_timestamp": group.get("creationTimestamp", "")
                                })
                                
                    except Exception as region_error:
                        logger.warning(f"Error getting instance groups from region {region_name}: {region_error}")
                        continue
            
            logger.info(f"✅ Found {len(all_groups)} managed instance groups")
            
            return {
                "total": len(all_groups),
                "groups": all_groups,
                "api_calls": [f"compute.instanceGroupManagers.list in {len(regions_response.get('items', []))} regions"]
            }
            
        except Exception as e:
            logger.error(f"Error getting instance groups: {e}")
            return {"total": 0, "groups": [], "error": str(e)}
    
    async def _get_load_balancers(self) -> Dict[str, Any]:
        """Get load balancers (forwarding rules)."""
        try:
            regional_lb = []
            global_lb = []
            
            # Get global forwarding rules
            try:
                global_response = self.gcp_service.call_google_api(
                    service="compute", 
                    version="v1",
                    resource_path=f"projects/{self.project_id}/global/forwardingRules"
                )
                
                if "items" in global_response:
                    for rule in global_response["items"]:
                        global_lb.append({
                            "name": rule.get("name", "unknown"),
                            "ip_address": rule.get("IPAddress", ""),
                            "port_range": rule.get("portRange", ""),
                            "target": rule.get("target", "").split("/")[-1] if rule.get("target") else None
                        })
                        
            except Exception as global_error:
                logger.warning(f"Error getting global forwarding rules: {global_error}")
            
            # Get regional forwarding rules
            try:
                regions_response = self.gcp_service.call_google_api(
                    service="compute",
                    version="v1", 
                    resource_path=f"projects/{self.project_id}/regions"
                )
                
                if "items" in regions_response:
                    for region in regions_response["items"][:3]:  # Limit for performance
                        region_name = region["name"]
                        try:
                            regional_response = self.gcp_service.call_google_api(
                                service="compute",
                                version="v1",
                                resource_path=f"projects/{self.project_id}/regions/{region_name}/forwardingRules"
                            )
                            
                            if "items" in regional_response:
                                for rule in regional_response["items"]:
                                    regional_lb.append({
                                        "name": rule.get("name", "unknown"),
                                        "region": region_name,
                                        "ip_address": rule.get("IPAddress", ""),
                                        "port_range": rule.get("portRange", ""),
                                        "target": rule.get("target", "").split("/")[-1] if rule.get("target") else None
                                    })
                                    
                        except Exception as region_error:
                            logger.warning(f"Error getting forwarding rules from region {region_name}: {region_error}")
                            continue
                            
            except Exception as regional_error:
                logger.warning(f"Error getting regional forwarding rules: {regional_error}")
            
            total_lb = len(regional_lb) + len(global_lb)
            logger.info(f"✅ Found {total_lb} load balancers ({len(regional_lb)} regional, {len(global_lb)} global)")
            
            return {
                "total": total_lb,
                "regional": regional_lb,
                "global": global_lb,
                "api_calls": ["compute.globalForwardingRules.list", "compute.forwardingRules.list"]
            }
            
        except Exception as e:
            logger.error(f"Error getting load balancers: {e}")
            return {"total": 0, "regional": [], "global": [], "error": str(e)}
    
    async def _get_app_engine_info(self) -> Dict[str, Any]:
        """Get App Engine application info."""
        try:
            # Use App Engine Admin API
            app_response = self.gcp_service.call_google_api(
                service="appengine",
                version="v1",
                resource_path=f"apps/{self.project_id}"
            )
            
            logger.info(f"✅ Found App Engine application: {app_response.get('id', 'unknown')}")
            
            return {
                "exists": True,
                "id": app_response.get("id", "unknown"),
                "location": app_response.get("locationId", "unknown"),
                "serving_status": app_response.get("servingStatus", "unknown"),
                "runtime": "standard",  # Most common
                "api_calls": ["appengine.apps.get"]
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                logger.info("No App Engine application found")
                return {"exists": False, "api_calls": ["appengine.apps.get"]}
            else:
                logger.warning(f"Error checking App Engine: {e}")
                return {"exists": False, "error": str(e), "api_calls": ["appengine.apps.get"]}
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback data if APIs fail."""
        return {
            "vm_instances": {"total": 0, "running": 0, "stopped": 0, "instances": [], "error": "API unavailable"},
            "instance_groups": {"total": 0, "groups": [], "error": "API unavailable"},
            "load_balancers": {"total": 0, "regional": [], "global": [], "error": "API unavailable"},
            "app_engine": {"exists": False, "error": "API unavailable"}
        }
    
    def format_inventory_response(self, inventory: Dict[str, Any]) -> str:
        """Format inventory data for chat response."""
        vm_data = inventory.get("vm_instances", {})
        ig_data = inventory.get("instance_groups", {})
        lb_data = inventory.get("load_balancers", {})
        ae_data = inventory.get("app_engine", {})
        
        # Collect all API calls made
        api_calls = []
        for data in [vm_data, ig_data, lb_data, ae_data]:
            if isinstance(data, dict) and "api_calls" in data:
                api_calls.extend(data["api_calls"])
        
        response = f"""📦 **Live Asset Inventory for {self.project_id}**

🖥️ **Compute Resources:**
• **VM Instances**: {vm_data.get('total', 0)} ({vm_data.get('running', 0)} running, {vm_data.get('stopped', 0)} stopped)
• **Instance Groups**: {ig_data.get('total', 0)} managed groups
• **Load Balancers**: {lb_data.get('total', 0)} load balancers
• **App Engine**: {"1 application" if ae_data.get('exists') else "No application deployed"}

🔍 **Data Source**: Live Google Cloud REST APIs
📡 **API Calls Made**:"""

        for api_call in set(api_calls):  # Remove duplicates
            response += f"\n• {api_call}"
        
        if vm_data.get('instances'):
            response += "\n\n🖥️ **Recent VM Instances:**"
            for instance in vm_data['instances'][:5]:
                response += f"\n• `{instance['name']}` ({instance['status']}) in {instance['zone']}"
        
        # Show any errors encountered
        errors = []
        for data_type, data in [("VM", vm_data), ("IG", ig_data), ("LB", lb_data), ("AE", ae_data)]:
            if isinstance(data, dict) and "error" in data:
                errors.append(f"{data_type}: {data['error']}")
        
        if errors:
            response += f"\n\n⚠️ **API Errors Encountered:**"
            for error in errors:
                response += f"\n• {error}"
        
        return response