"""
Connectivity Testing API Endpoints
==================================

FastAPI endpoints for network connectivity testing functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.connectivity_tester import ConnectivityTester
from ..models.network_models import (
    ConnectivityTestResult, ConnectivityTestRequest, NetworkEndpoint,
    TestType, TestStatus, create_network_endpoint
)

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1/networking/connectivity", tags=["connectivity"])

# Global connectivity tester instance
connectivity_tester = ConnectivityTester()


# Request/Response Models
class ConnectivityTestResponse(BaseModel):
    """Response model for connectivity test requests"""
    test_id: str
    status: str
    message: str
    results: List[Dict[str, Any]] = Field(default_factory=list)


class TestHistoryResponse(BaseModel):
    """Response model for test history requests"""
    total_tests: int
    tests: List[Dict[str, Any]]
    filters_applied: Dict[str, Any]


# API Endpoints

@router.post("/test", response_model=ConnectivityTestResponse)
async def run_connectivity_test(
    request: ConnectivityTestRequest,
    background_tasks: BackgroundTasks
):
    """
    Run comprehensive connectivity test
    
    Args:
        request: Connectivity test request with source, destination, and test types
        background_tasks: Background task manager for async execution
        
    Returns:
        Test response with results or test ID for async operations
    """
    try:
        logger.info(f"Received connectivity test request: {request.destination.ip_address}")
        
        # Validate request
        if not request.destination.ip_address:
            raise HTTPException(status_code=400, detail="Destination IP address is required")
        
        results = []
        
        # Run requested tests
        for test_type in request.test_types:
            try:
                if test_type == TestType.PING:
                    result = await connectivity_tester.ping_test(
                        destination=request.destination,
                        timeout=min(request.timeout_seconds, 30)  # Max 30 seconds for ping
                    )
                    results.append(result.to_dict())
                
                elif test_type == TestType.TCP_CONNECT:
                    if not request.destination.port:
                        raise HTTPException(
                            status_code=400, 
                            detail="Port is required for TCP connectivity test"
                        )
                    result = await connectivity_tester.port_connectivity_test(
                        destination=request.destination,
                        timeout=request.timeout_seconds
                    )
                    results.append(result.to_dict())
                
                elif test_type == TestType.TRACEROUTE:
                    result = await connectivity_tester.traceroute_test(
                        destination=request.destination,
                        timeout=request.timeout_seconds
                    )
                    results.append(result.to_dict())
                
                elif test_type == TestType.HTTP_CHECK:
                    # TODO: Implement HTTP check
                    raise HTTPException(
                        status_code=501, 
                        detail="HTTP connectivity test not yet implemented"
                    )
                
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported test type: {test_type}"
                    )
                    
            except Exception as test_error:
                logger.error(f"Error in {test_type} test: {test_error}")
                # Create failed result
                failed_result = ConnectivityTestResult(
                    test_id=f"failed_{test_type.value}",
                    source=create_network_endpoint("127.0.0.1"),
                    destination=request.destination,
                    test_type=test_type,
                    status=TestStatus.FAILURE,
                    error_message=str(test_error),
                    timestamp=datetime.now(),
                    duration_ms=0
                )
                results.append(failed_result.to_dict())
        
        # Determine overall status
        success_count = sum(1 for result in results if result.get('is_successful', False))
        total_count = len(results)
        
        if success_count == total_count:
            overall_status = "SUCCESS"
            message = f"All {total_count} connectivity tests passed"
        elif success_count == 0:
            overall_status = "FAILURE" 
            message = f"All {total_count} connectivity tests failed"
        else:
            overall_status = "PARTIAL"
            message = f"{success_count}/{total_count} connectivity tests passed"
        
        return ConnectivityTestResponse(
            test_id=results[0]['test_id'] if results else "no_tests",
            status=overall_status,
            message=message,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in connectivity test: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/quick-ping/{ip_address}")
async def quick_ping_test(ip_address: str):
    """
    Quick ping test to an IP address
    
    Args:
        ip_address: IP address to ping
        
    Returns:
        Simple ping test result
    """
    try:
        logger.info(f"Quick ping test to {ip_address}")
        
        # Validate IP format (basic check)
        import socket
        try:
            socket.inet_aton(ip_address)
        except socket.error:
            raise HTTPException(status_code=400, detail="Invalid IP address format")
        
        destination = create_network_endpoint(ip_address)
        result = await connectivity_tester.ping_test(destination, count=2, timeout=5)
        
        return {
            "ip_address": ip_address,
            "reachable": result.is_successful,
            "latency_ms": result.latency_ms,
            "packet_loss_percent": result.packet_loss_percent,
            "test_id": result.test_id,
            "timestamp": result.timestamp.isoformat(),
            "error_message": result.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick ping test: {e}")
        raise HTTPException(status_code=500, detail=f"Ping test failed: {str(e)}")


@router.get("/history", response_model=TestHistoryResponse)
async def get_connectivity_history(
    destination_ip: Optional[str] = None,
    test_type: Optional[str] = None,
    limit: int = 50
):
    """
    Get connectivity test history
    
    Args:
        destination_ip: Filter by destination IP (optional)
        test_type: Filter by test type (optional)
        limit: Maximum number of results (default: 50, max: 500)
        
    Returns:
        Historical test results
    """
    try:
        # Validate and limit the query
        limit = min(max(1, limit), 500)  # Between 1 and 500
        
        test_type_enum = None
        if test_type:
            try:
                test_type_enum = TestType(test_type.upper())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid test type: {test_type}")
        
        logger.info(f"Retrieving connectivity history (limit: {limit})")
        
        # Get test history from database
        results = await connectivity_tester.get_test_history(
            destination_ip=destination_ip,
            test_type=test_type_enum,
            limit=limit
        )
        
        # Convert results to dictionaries
        test_data = [result.to_dict() for result in results]
        
        filters_applied = {}
        if destination_ip:
            filters_applied["destination_ip"] = destination_ip
        if test_type:
            filters_applied["test_type"] = test_type
        filters_applied["limit"] = limit
        
        return TestHistoryResponse(
            total_tests=len(test_data),
            tests=test_data,
            filters_applied=filters_applied
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving test history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.get("/status/{test_id}")
async def get_test_status(test_id: str):
    """
    Get status of a specific connectivity test
    
    Args:
        test_id: Test ID to look up
        
    Returns:
        Test result details
    """
    try:
        logger.info(f"Looking up test status for: {test_id}")
        
        result = await connectivity_tester.get_test_status(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Test ID not found: {test_id}")
        
        return {
            "found": True,
            "test_result": result.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test status: {str(e)}")


@router.post("/batch-test")
async def batch_connectivity_test(
    destinations: List[str],
    test_types: List[str] = ["PING"],
    timeout_seconds: int = 30
):
    """
    Run connectivity tests to multiple destinations
    
    Args:
        destinations: List of IP addresses to test
        test_types: List of test types to run
        timeout_seconds: Timeout for each test
        
    Returns:
        Results for all destination tests
    """
    try:
        if len(destinations) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 destinations allowed for batch test")
        
        # Validate test types
        valid_test_types = []
        for test_type in test_types:
            try:
                valid_test_types.append(TestType(test_type.upper()))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid test type: {test_type}")
        
        logger.info(f"Running batch connectivity test to {len(destinations)} destinations")
        
        all_results = []
        
        # Run tests concurrently for better performance
        async def test_destination(ip_address: str):
            try:
                destination = create_network_endpoint(ip_address)
                destination_results = []
                
                for test_type in valid_test_types:
                    if test_type == TestType.PING:
                        result = await connectivity_tester.ping_test(destination, timeout=min(timeout_seconds, 10))
                    elif test_type == TestType.TCP_CONNECT:
                        # Use common ports if no port specified
                        destination.port = 80  # Default to HTTP port
                        result = await connectivity_tester.port_connectivity_test(destination, timeout=timeout_seconds)
                    else:
                        continue  # Skip unsupported test types for batch
                    
                    destination_results.append({
                        "destination": ip_address,
                        "test_type": test_type.value,
                        "result": result.to_dict()
                    })
                
                return destination_results
                
            except Exception as e:
                logger.error(f"Error testing destination {ip_address}: {e}")
                return [{
                    "destination": ip_address,
                    "test_type": "ERROR",
                    "result": {"error": str(e)}
                }]
        
        # Run all destination tests concurrently
        tasks = [test_destination(ip) for ip in destinations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for result in results:
            if isinstance(result, Exception):
                all_results.append({"error": str(result)})
            else:
                all_results.extend(result)
        
        # Calculate summary statistics
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if r.get('result', {}).get('is_successful', False))
        
        return {
            "batch_summary": {
                "total_destinations": len(destinations),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "results": all_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch connectivity test: {e}")
        raise HTTPException(status_code=500, detail=f"Batch test failed: {str(e)}")


@router.get("/health")
async def connectivity_service_health():
    """Health check endpoint for connectivity testing service"""
    try:
        # Test database connectivity
        test_history = await connectivity_tester.get_test_history(limit=1)
        
        return {
            "status": "healthy",
            "service": "connectivity_tester",
            "database": "accessible",
            "timestamp": datetime.now().isoformat(),
            "recent_tests": len(test_history)
        }
        
    except Exception as e:
        logger.error(f"Connectivity service health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Export router
__all__ = ["router"]