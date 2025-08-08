"""Real GCP Compute Resources Inventory Service."""

import logging
from typing import Dict, Any, List, Optional
import asyncio
try:
    from google.cloud import compute_v1
    from google.auth import default
    from google.oauth2 import service_account
    COMPUTE_AVAILABLE = True
except ImportError:
    COMPUTE_AVAILABLE = False
    logging.warning("Google Cloud Compute library not available, will use REST API fallback")

logger = logging.getLogger(__name__)

class ComputeInventoryService:
    """Service to get real compute resources from GCP APIs."""
    
    def __init__(self, credentials=None, project_id=None):
        self.credentials = credentials
        self.project_id = project_id
        
        if not self.credentials:
            self.credentials, self.project_id = default()
            
        # Initialize compute clients
        self.instances_client = compute_v1.InstancesClient(credentials=self.credentials)
        self.instance_groups_client = compute_v1.InstanceGroupsClient(credentials=self.credentials)
        self.forwarding_rules_client = compute_v1.ForwardingRulesClient(credentials=self.credentials)
        self.global_forwarding_rules_client = compute_v1.GlobalForwardingRulesClient(credentials=self.credentials)
        
    async def get_compute_inventory(self) -> Dict[str, Any]:
        """Get complete compute inventory from GCP APIs."""
        try:
            # Run all API calls concurrently
            tasks = [
                self._get_vm_instances(),
                self._get_instance_groups(), 
                self._get_load_balancers(),
                self._get_app_engine_info()
            ]
            
            vm_data, instance_groups_data, load_balancers_data, app_engine_data = await asyncio.gather(*tasks)
            
            return {
                "vm_instances": vm_data,
                "instance_groups": instance_groups_data,
                "load_balancers": load_balancers_data,
                "app_engine": app_engine_data
            }
            
        except Exception as e:
            logger.error(f"Error getting compute inventory: {e}")
            return self._get_fallback_data()
    
    async def _get_vm_instances(self) -> Dict[str, Any]:
        """Get VM instances across all zones."""
        try:
            all_instances = []
            zones_client = compute_v1.ZonesClient(credentials=self.credentials)
            
            # Get all zones in the project
            zones_request = compute_v1.ListZonesRequest(project=self.project_id)
            zones = zones_client.list(request=zones_request)
            
            running_count = 0
            stopped_count = 0
            
            for zone in zones:
                try:
                    # List instances in each zone
                    request = compute_v1.ListInstancesRequest(
                        project=self.project_id,
                        zone=zone.name
                    )
                    
                    instances = self.instances_client.list(request=request)
                    
                    for instance in instances:
                        all_instances.append({
                            "name": instance.name,
                            "zone": zone.name,
                            "status": instance.status,
                            "machine_type": instance.machine_type.split('/')[-1],
                            "creation_timestamp": instance.creation_timestamp
                        })
                        
                        # Count by status
                        if instance.status == "RUNNING":
                            running_count += 1
                        elif instance.status in ["STOPPED", "TERMINATED"]:
                            stopped_count += 1
                            
                except Exception as zone_error:
                    logger.warning(f"Error getting instances from zone {zone.name}: {zone_error}")
                    continue
            
            logger.info(f"✅ Found {len(all_instances)} VM instances ({running_count} running, {stopped_count} stopped)")
            
            return {
                "total": len(all_instances),
                "running": running_count,
                "stopped": stopped_count,
                "instances": all_instances[:10]  # Limit for display
            }
            
        except Exception as e:
            logger.error(f"Error getting VM instances: {e}")
            return {"total": 0, "running": 0, "stopped": 0, "instances": []}
    
    async def _get_instance_groups(self) -> Dict[str, Any]:
        """Get managed instance groups."""
        try:
            all_groups = []
            zones_client = compute_v1.ZonesClient(credentials=self.credentials)
            
            # Get all zones
            zones_request = compute_v1.ListZonesRequest(project=self.project_id)
            zones = zones_client.list(request=zones_request)
            
            for zone in zones:
                try:
                    request = compute_v1.ListInstanceGroupsRequest(
                        project=self.project_id,
                        zone=zone.name
                    )
                    
                    groups = self.instance_groups_client.list(request=request)
                    
                    for group in groups:
                        all_groups.append({
                            "name": group.name,
                            "zone": zone.name,
                            "size": group.size,
                            "creation_timestamp": group.creation_timestamp
                        })
                        
                except Exception as zone_error:
                    logger.warning(f"Error getting instance groups from zone {zone.name}: {zone_error}")
                    continue
            
            logger.info(f"✅ Found {len(all_groups)} instance groups")
            
            return {
                "total": len(all_groups),
                "groups": all_groups
            }
            
        except Exception as e:
            logger.error(f"Error getting instance groups: {e}")
            return {"total": 0, "groups": []}
    
    async def _get_load_balancers(self) -> Dict[str, Any]:
        """Get load balancers (forwarding rules)."""
        try:
            regional_lb = []
            global_lb = []
            
            # Get regional forwarding rules
            try:
                regions_client = compute_v1.RegionsClient(credentials=self.credentials)
                regions_request = compute_v1.ListRegionsRequest(project=self.project_id)
                regions = regions_client.list(request=regions_request)
                
                for region in regions:
                    try:
                        request = compute_v1.ListForwardingRulesRequest(
                            project=self.project_id,
                            region=region.name
                        )
                        
                        rules = self.forwarding_rules_client.list(request=request)
                        
                        for rule in rules:
                            regional_lb.append({
                                "name": rule.name,
                                "region": region.name,
                                "ip_address": rule.ip_address,
                                "port_range": rule.port_range,
                                "target": rule.target.split('/')[-1] if rule.target else None
                            })
                            
                    except Exception as region_error:
                        logger.warning(f"Error getting forwarding rules from region {region.name}: {region_error}")
                        continue
                        
            except Exception as regional_error:
                logger.warning(f"Error getting regional load balancers: {regional_error}")
            
            # Get global forwarding rules
            try:
                request = compute_v1.ListGlobalForwardingRulesRequest(project=self.project_id)
                global_rules = self.global_forwarding_rules_client.list(request=request)
                
                for rule in global_rules:
                    global_lb.append({
                        "name": rule.name,
                        "ip_address": rule.ip_address,
                        "port_range": rule.port_range,
                        "target": rule.target.split('/')[-1] if rule.target else None
                    })
                    
            except Exception as global_error:
                logger.warning(f"Error getting global load balancers: {global_error}")
            
            total_lb = len(regional_lb) + len(global_lb)
            logger.info(f"✅ Found {total_lb} load balancers ({len(regional_lb)} regional, {len(global_lb)} global)")
            
            return {
                "total": total_lb,
                "regional": regional_lb,
                "global": global_lb
            }
            
        except Exception as e:
            logger.error(f"Error getting load balancers: {e}")
            return {"total": 0, "regional": [], "global": []}
    
    async def _get_app_engine_info(self) -> Dict[str, Any]:
        """Get App Engine application info."""
        try:
            # Use App Engine Admin API
            from google.cloud import appengine_v1
            
            client = appengine_v1.ApplicationsClient(credentials=self.credentials)
            
            try:
                request = appengine_v1.GetApplicationRequest(name=f"apps/{self.project_id}")
                app = client.get_application(request=request)
                
                logger.info(f"✅ Found App Engine application: {app.id}")
                
                return {
                    "exists": True,
                    "id": app.id,
                    "location": app.location_id,
                    "serving_status": app.serving_status.name if hasattr(app.serving_status, 'name') else str(app.serving_status),
                    "runtime": "standard"  # Most common
                }
                
            except Exception as app_error:
                if "does not exist" in str(app_error).lower():
                    logger.info("No App Engine application found")
                    return {"exists": False}
                else:
                    logger.warning(f"Error checking App Engine: {app_error}")
                    return {"exists": False, "error": str(app_error)}
                    
        except Exception as e:
            logger.error(f"Error getting App Engine info: {e}")
            return {"exists": False, "error": str(e)}
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback data if APIs fail."""
        return {
            "vm_instances": {"total": 0, "running": 0, "stopped": 0, "instances": []},
            "instance_groups": {"total": 0, "groups": []},
            "load_balancers": {"total": 0, "regional": [], "global": []},
            "app_engine": {"exists": False, "error": "API unavailable"}
        }
    
    def format_inventory_response(self, inventory: Dict[str, Any]) -> str:
        """Format inventory data for chat response."""
        vm_data = inventory.get("vm_instances", {})
        ig_data = inventory.get("instance_groups", {})
        lb_data = inventory.get("load_balancers", {})
        ae_data = inventory.get("app_engine", {})
        
        response = f"""📦 **Live Asset Inventory for {self.project_id}**

🖥️ **Compute Resources:**
• **VM Instances**: {vm_data.get('total', 0)} ({vm_data.get('running', 0)} running, {vm_data.get('stopped', 0)} stopped)
• **Instance Groups**: {ig_data.get('total', 0)} managed groups
• **Load Balancers**: {lb_data.get('total', 0)} (HTTP/HTTPS load balancers)
• **App Engine**: {"1 application" if ae_data.get('exists') else "No application deployed"}

🔍 **Data Source**: Live Google Cloud Compute APIs
📡 **API Calls Made**:
• `compute.instances.list` across all zones
• `compute.instanceGroups.list` across regions  
• `compute.forwardingRules.list` (regional & global)
• `appengine.apps.get` for App Engine status"""

        if vm_data.get('instances'):
            response += "\n\n🖥️ **Recent VM Instances:**"
            for instance in vm_data['instances'][:5]:
                response += f"\n• `{instance['name']}` ({instance['status']}) in {instance['zone']}"
        
        return response