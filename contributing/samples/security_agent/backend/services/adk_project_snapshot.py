#!/usr/bin/env python3
"""
🔄 ADK Project Snapshot Service

Following ADK agent patterns for efficient GCP resource caching.
Reduces API calls by maintaining intelligent project snapshots.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ProjectSnapshot:
    """ADK-compliant project snapshot data structure."""
    project_id: str
    snapshot_time: float
    cache_ttl: int = 3600  # 1 hour default
    asset_count: int = 0
    bucket_count: int = 0
    compute_instances: int = 0
    storage_size_gb: float = 0.0
    billing_info: Dict[str, Any] = None
    security_findings: List[Dict[str, Any]] = None
    recommendations: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.billing_info is None:
            self.billing_info = {}
        if self.security_findings is None:
            self.security_findings = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if snapshot is expired using ADK TTL pattern."""
        return (time.time() - self.snapshot_time) > self.cache_ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class ADKProjectSnapshotService:
    """
    ADK-compliant project snapshot service for efficient resource caching.
    
    Implements ADK patterns:
    - Persistent state management
    - Intelligent caching with TTL
    - Graceful API fallbacks
    - Performance optimization
    """
    
    def __init__(self, cache_dir: str = "cache/snapshots"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: Dict[str, ProjectSnapshot] = {}
        logger.info(f"🔄 ADK Snapshot Service initialized with cache: {self.cache_dir}")
    
    def _get_cache_path(self, project_id: str) -> Path:
        """Get cache file path for a project."""
        return self.cache_dir / f"{project_id}_snapshot.json"
    
    def _generate_cache_key(self, project_id: str, force_refresh: bool = False) -> str:
        """Generate cache key using ADK patterns."""
        base_key = f"adk_snapshot_{project_id}"
        if force_refresh:
            base_key += f"_refresh_{int(time.time())}"
        return hashlib.sha256(base_key.encode()).hexdigest()[:16]
    
    async def get_project_snapshot(
        self, 
        project_id: str, 
        force_refresh: bool = False
    ) -> ProjectSnapshot:
        """
        Get project snapshot with intelligent caching.
        
        ADK pattern: Check cache first, fetch if expired/missing.
        """
        logger.info(f"📸 ADK: Getting snapshot for project {project_id} (refresh={force_refresh})")
        
        # Check memory cache first
        if not force_refresh and project_id in self.snapshots:
            snapshot = self.snapshots[project_id]
            if not snapshot.is_expired():
                logger.info(f"✅ ADK: Using cached snapshot (age: {int(time.time() - snapshot.snapshot_time)}s)")
                return snapshot
        
        # Check disk cache
        cache_path = self._get_cache_path(project_id)
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    snapshot_data = json.load(f)
                snapshot = ProjectSnapshot(**snapshot_data)
                
                if not snapshot.is_expired():
                    self.snapshots[project_id] = snapshot
                    logger.info(f"✅ ADK: Loaded snapshot from disk (age: {int(time.time() - snapshot.snapshot_time)}s)")
                    return snapshot
            except Exception as e:
                logger.warning(f"⚠️ ADK: Failed to load cached snapshot: {e}")
        
        # Generate new snapshot
        logger.info(f"🔄 ADK: Generating fresh snapshot for {project_id}")
        snapshot = await self._generate_snapshot(project_id)
        
        # Cache in memory and disk
        self.snapshots[project_id] = snapshot
        await self._save_snapshot(snapshot)
        
        return snapshot
    
    async def _generate_snapshot(self, project_id: str) -> ProjectSnapshot:
        """Generate fresh project snapshot using ADK service patterns."""
        start_time = time.time()
        
        snapshot = ProjectSnapshot(
            project_id=project_id,
            snapshot_time=start_time,
            cache_ttl=3600,  # 1 hour TTL
            metadata={
                "generated_by": "adk_snapshot_service",
                "version": "1.0",
                "adk_compliant": True
            }
        )
        
        try:
            # Use existing enhanced asset inventory service
            from .enhanced_asset_inventory_service import EnhancedAssetInventoryService
            asset_service = EnhancedAssetInventoryService()
            
            logger.info(f"📊 ADK: Fetching asset inventory for {project_id}")
            assets = await asset_service.get_asset_inventory_async(
                project_id, 
                use_cache=True,
                cache_ttl=1800  # 30 minutes for assets
            )
            
            # Process asset data
            if assets and assets.get("success"):
                asset_data = assets.get("assets", [])
                snapshot.asset_count = len(asset_data)
                
                # Count buckets
                buckets = [a for a in asset_data if 'storage.googleapis.com/Bucket' in a.get('asset_type', '')]
                snapshot.bucket_count = len(buckets)
                
                # Count compute instances
                instances = [a for a in asset_data if 'compute.googleapis.com/Instance' in a.get('asset_type', '')]
                snapshot.compute_instances = len(instances)
                
                # Estimate storage size
                snapshot.storage_size_gb = len(buckets) * 10.5  # Rough estimate
                
                logger.info(f"📊 ADK: Snapshot generated - {snapshot.asset_count} assets, {snapshot.bucket_count} buckets")
            
            # Get basic billing info (if available)
            snapshot.billing_info = {
                "enabled": True,
                "estimated_monthly": f"${(snapshot.asset_count * 0.50):.2f}",
                "last_updated": datetime.now().isoformat()
            }
            
            # Generate proactive recommendations using ADK patterns
            snapshot.recommendations = await self._generate_proactive_recommendations(project_id, snapshot)
            
            generation_time = time.time() - start_time
            snapshot.metadata["generation_time_ms"] = round(generation_time * 1000, 2)
            
            logger.info(f"✅ ADK: Snapshot generated in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ ADK: Failed to generate snapshot: {e}")
            # Return basic snapshot even on failure
            snapshot.metadata["error"] = str(e)
            snapshot.metadata["fallback_mode"] = True
        
        return snapshot
    
    async def _generate_proactive_recommendations(
        self, 
        project_id: str, 
        snapshot: ProjectSnapshot
    ) -> List[Dict[str, Any]]:
        """Generate proactive recommendations using ADK agent patterns."""
        recommendations = []
        
        try:
            # Storage recommendations
            if snapshot.bucket_count > 0:
                recommendations.append({
                    "type": "storage_security",
                    "priority": "high",
                    "title": "Enable Uniform Bucket-Level Access",
                    "description": f"Consider enabling uniform bucket-level access for your {snapshot.bucket_count} storage buckets",
                    "action": "Review bucket permissions and enable uniform access",
                    "estimated_effort": "15 minutes",
                    "impact": "Improved security posture"
                })
            
            # Compute recommendations
            if snapshot.compute_instances > 5:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "medium", 
                    "title": "Optimize Compute Usage",
                    "description": f"You have {snapshot.compute_instances} compute instances running",
                    "action": "Review usage patterns and consider rightsizing",
                    "estimated_effort": "30 minutes",
                    "impact": "Potential cost savings"
                })
            
            # Generic security recommendation
            if snapshot.asset_count > 10:
                recommendations.append({
                    "type": "security_monitoring",
                    "priority": "medium",
                    "title": "Enable Security Command Center", 
                    "description": f"Monitor {snapshot.asset_count} assets with centralized security insights",
                    "action": "Enable Security Command Center for comprehensive monitoring",
                    "estimated_effort": "10 minutes",
                    "impact": "Enhanced security visibility"
                })
            
            logger.info(f"💡 ADK: Generated {len(recommendations)} proactive recommendations")
            
        except Exception as e:
            logger.error(f"❌ ADK: Failed to generate recommendations: {e}")
        
        return recommendations
    
    async def _save_snapshot(self, snapshot: ProjectSnapshot):
        """Save snapshot to disk using ADK persistence patterns."""
        try:
            cache_path = self._get_cache_path(snapshot.project_id)
            with open(cache_path, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            logger.info(f"💾 ADK: Saved snapshot to {cache_path}")
        except Exception as e:
            logger.error(f"❌ ADK: Failed to save snapshot: {e}")
    
    async def get_multiple_snapshots(
        self, 
        project_ids: List[str], 
        force_refresh: bool = False
    ) -> Dict[str, ProjectSnapshot]:
        """Get multiple project snapshots concurrently."""
        logger.info(f"📸 ADK: Getting snapshots for {len(project_ids)} projects")
        
        tasks = [
            self.get_project_snapshot(pid, force_refresh) 
            for pid in project_ids
        ]
        
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for i, snapshot in enumerate(snapshots):
            if isinstance(snapshot, Exception):
                logger.error(f"❌ ADK: Failed to get snapshot for {project_ids[i]}: {snapshot}")
            else:
                result[project_ids[i]] = snapshot
        
        return result
    
    def get_cached_recommendations(self, project_id: str) -> List[Dict[str, Any]]:
        """Get cached recommendations without API calls."""
        if project_id in self.snapshots:
            return self.snapshots[project_id].recommendations
        
        # Try disk cache
        cache_path = self._get_cache_path(project_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    snapshot_data = json.load(f)
                return snapshot_data.get('recommendations', [])
            except Exception as e:
                logger.warning(f"⚠️ ADK: Failed to load cached recommendations: {e}")
        
        return []
    
    def cleanup_expired_snapshots(self):
        """Clean up expired snapshots from memory and disk."""
        expired_projects = []
        
        # Clean memory cache
        for project_id, snapshot in self.snapshots.items():
            if snapshot.is_expired():
                expired_projects.append(project_id)
        
        for project_id in expired_projects:
            del self.snapshots[project_id]
        
        # Clean disk cache
        for cache_file in self.cache_dir.glob("*_snapshot.json"):
            try:
                age = time.time() - cache_file.stat().st_mtime
                if age > 7200:  # 2 hours
                    cache_file.unlink()
                    logger.info(f"🧹 ADK: Cleaned expired cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"⚠️ ADK: Failed to clean cache file {cache_file}: {e}")
        
        if expired_projects:
            logger.info(f"🧹 ADK: Cleaned {len(expired_projects)} expired snapshots")

# Global service instance following ADK singleton pattern
_snapshot_service: Optional[ADKProjectSnapshotService] = None

def get_snapshot_service() -> ADKProjectSnapshotService:
    """Get global snapshot service instance (ADK singleton pattern)."""
    global _snapshot_service
    if _snapshot_service is None:
        _snapshot_service = ADKProjectSnapshotService()
    return _snapshot_service

async def get_project_metrics(project_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Quick access function for project metrics."""
    service = get_snapshot_service()
    snapshot = await service.get_project_snapshot(project_id, force_refresh)
    
    return {
        "project_id": project_id,
        "asset_count": snapshot.asset_count,
        "bucket_count": snapshot.bucket_count,
        "compute_instances": snapshot.compute_instances,
        "storage_size_gb": snapshot.storage_size_gb,
        "recommendations_count": len(snapshot.recommendations),
        "cache_age_minutes": int((time.time() - snapshot.snapshot_time) / 60),
        "is_cached": not snapshot.metadata.get("fallback_mode", False)
    }