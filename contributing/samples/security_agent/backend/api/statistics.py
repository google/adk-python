"""
Statistical Analysis API Endpoints (STORY-006)
==============================================

Provides REST API endpoints for statistical analysis including:
- Trend analysis
- Anomaly detection
- Correlation analysis
- Forecasting
- Pattern recognition
- Automated insights
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

# Import the statistical analyzer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from services.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/statistics",
    tags=["statistics"],
    responses={404: {"description": "Not found"}},
)

# Initialize analyzer
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
analyzer = StatisticalAnalyzer(database_path)

# Pydantic models for requests/responses
class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""
    metric_type: str = Field(..., description="Type of metric to analyze")
    metric_column: str = Field(..., description="Column name for analysis")
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")

class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    metric_type: str = Field(..., description="Type of metric to analyze")
    metric_column: str = Field(..., description="Column name for analysis")
    sensitivity: float = Field(2.0, ge=1.0, le=5.0, description="Sensitivity level")
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")

class CorrelationRequest(BaseModel):
    """Request model for correlation analysis."""
    metrics: List[str] = Field(..., min_items=2, description="List of metrics to correlate")
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")

class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    metric_type: str = Field(..., description="Type of metric to forecast")
    metric_column: str = Field(..., description="Column name for forecasting")
    horizon: int = Field(7, ge=1, le=30, description="Forecast horizon in days")
    days: int = Field(30, ge=7, le=365, description="Historical days for training")

class ComprehensiveAnalysisRequest(BaseModel):
    """Request model for comprehensive analysis."""
    metric_types: Optional[List[str]] = Field(None, description="Metrics to analyze")
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")

class StatisticsResponse(BaseModel):
    """Response model for statistical analysis."""
    success: bool
    data: Dict[str, Any]
    timestamp: str
    message: Optional[str] = None

@router.get("/health")
async def health_check():
    """Check statistics service health."""
    return {
        "status": "healthy",
        "service": "statistics",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/trends", response_model=StatisticsResponse)
async def analyze_trends(request: TrendAnalysisRequest):
    """Analyze trends for a specific metric."""
    try:
        logger.info(f"Analyzing trends for {request.metric_type}/{request.metric_column}")
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        data = analyzer.get_metrics_data(request.metric_type, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Calculate trends
        trends = analyzer.calculate_trends(data, request.metric_column)
        
        if 'error' in trends:
            raise HTTPException(status_code=400, detail=trends['error'])
        
        return StatisticsResponse(
            success=True,
            data=trends,
            timestamp=datetime.now().isoformat(),
            message=f"Trend analysis completed for {request.metric_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/anomalies", response_model=StatisticsResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in metric data."""
    try:
        logger.info(f"Detecting anomalies in {request.metric_type}/{request.metric_column}")
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        data = analyzer.get_metrics_data(request.metric_type, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(data, request.metric_column, request.sensitivity)
        
        if 'error' in anomalies:
            raise HTTPException(status_code=400, detail=anomalies['error'])
        
        return StatisticsResponse(
            success=True,
            data=anomalies,
            timestamp=datetime.now().isoformat(),
            message=f"Detected {anomalies.get('total_anomalies', 0)} anomalies"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlations", response_model=StatisticsResponse)
async def analyze_correlations(request: CorrelationRequest):
    """Analyze correlations between metrics."""
    try:
        logger.info(f"Analyzing correlations between {request.metrics}")
        
        # Get data for first metric type (simplified)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        
        # For correlation, we need to combine data from different sources
        # This is a simplified version - in production, you'd merge properly
        data = analyzer.get_metrics_data(request.metrics[0], start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Perform correlation analysis
        correlations = analyzer.correlation_analysis(data, request.metrics)
        
        if 'error' in correlations:
            raise HTTPException(status_code=400, detail=correlations['error'])
        
        return StatisticsResponse(
            success=True,
            data=correlations,
            timestamp=datetime.now().isoformat(),
            message=f"Correlation analysis completed for {len(request.metrics)} metrics"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast", response_model=StatisticsResponse)
async def forecast_metric(request: ForecastRequest):
    """Generate forecast for a metric."""
    try:
        logger.info(f"Forecasting {request.metric_type}/{request.metric_column} for {request.horizon} days")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        data = analyzer.get_metrics_data(request.metric_type, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for forecasting")
        
        # Generate forecast
        forecast = analyzer.forecast(data, request.metric_column, request.horizon)
        
        if 'error' in forecast:
            raise HTTPException(status_code=400, detail=forecast['error'])
        
        return StatisticsResponse(
            success=True,
            data=forecast,
            timestamp=datetime.now().isoformat(),
            message=f"Generated {request.horizon}-day forecast for {request.metric_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/patterns", response_model=StatisticsResponse)
async def identify_patterns(request: TrendAnalysisRequest):
    """Identify patterns in metric data."""
    try:
        logger.info(f"Identifying patterns in {request.metric_type}/{request.metric_column}")
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        data = analyzer.get_metrics_data(request.metric_type, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Identify patterns
        patterns = analyzer.pattern_recognition(data, request.metric_column)
        
        if 'error' in patterns:
            raise HTTPException(status_code=400, detail=patterns['error'])
        
        return StatisticsResponse(
            success=True,
            data=patterns,
            timestamp=datetime.now().isoformat(),
            message=f"Pattern analysis completed for {request.metric_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pattern recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comprehensive", response_model=StatisticsResponse)
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """Perform comprehensive statistical analysis."""
    try:
        logger.info(f"Running comprehensive analysis for {request.days} days")
        
        # Perform comprehensive analysis
        results = analyzer.comprehensive_analysis(
            metric_types=request.metric_types,
            days=request.days
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        return StatisticsResponse(
            success=True,
            data=results,
            timestamp=datetime.now().isoformat(),
            message=f"Comprehensive analysis completed: {results.get('summary', {})}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=StatisticsResponse)
async def get_insights(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get automated insights from statistical analysis."""
    try:
        logger.info(f"Generating insights for last {days} days")
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis(days=days)
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        # Extract just the insights
        insights_data = {
            'insights': results.get('insights', []),
            'summary': results.get('summary', {}),
            'period_days': days,
            'generated_at': datetime.now().isoformat()
        }
        
        return StatisticsResponse(
            success=True,
            data=insights_data,
            timestamp=datetime.now().isoformat(),
            message=f"Generated {len(insights_data['insights'])} insights"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/available")
async def get_available_metrics():
    """Get list of available metrics for analysis."""
    try:
        available_metrics = {
            'security_findings': {
                'description': 'Security findings and vulnerabilities',
                'columns': ['severity_score', 'count', 'risk_level'],
                'supported_analyses': ['trends', 'anomalies', 'forecast', 'patterns']
            },
            'iam_policies': {
                'description': 'IAM policies and permissions',
                'columns': ['member_count', 'permission_count', 'risk_score'],
                'supported_analyses': ['trends', 'anomalies', 'correlations']
            },
            'storage_buckets': {
                'description': 'Cloud storage bucket metrics',
                'columns': ['size_bytes', 'object_count', 'public_access_count'],
                'supported_analyses': ['trends', 'anomalies', 'forecast']
            },
            'firewall_rules': {
                'description': 'Firewall rules and network security',
                'columns': ['rule_count', 'open_ports', 'priority'],
                'supported_analyses': ['trends', 'patterns']
            },
            'api_keys': {
                'description': 'API key usage and security',
                'columns': ['usage_count', 'restriction_count', 'age_days'],
                'supported_analyses': ['trends', 'anomalies', 'forecast']
            },
            'recommendations': {
                'description': 'Security recommendations',
                'columns': ['priority_score', 'impact_score', 'count'],
                'supported_analyses': ['trends', 'correlations']
            }
        }
        
        return {
            'success': True,
            'metrics': available_metrics,
            'total_metrics': len(available_metrics),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/summary")
async def get_statistical_summary(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get statistical summary report."""
    try:
        logger.info(f"Generating statistical summary for {days} days")
        
        # Run analysis
        results = analyzer.comprehensive_analysis(days=days)
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        # Create summary report
        summary = {
            'period': f"Last {days} days",
            'generated_at': datetime.now().isoformat(),
            'key_metrics': {
                'total_anomalies': results.get('summary', {}).get('total_anomalies_detected', 0),
                'strong_correlations': results.get('summary', {}).get('strong_correlations_found', 0),
                'insights_generated': results.get('summary', {}).get('insights_generated', 0),
                'metrics_analyzed': results.get('summary', {}).get('total_metrics_analyzed', 0)
            },
            'top_insights': results.get('insights', [])[:5],
            'trend_summary': {
                metric: {
                    'direction': trend.get('trend_direction'),
                    'strength': trend.get('trend_strength'),
                    'current_value': trend.get('current_value')
                }
                for metric, trend in results.get('trends', {}).items()
                if isinstance(trend, dict) and 'trend_direction' in trend
            },
            'anomaly_summary': {
                metric: {
                    'total': anomaly.get('total_anomalies', 0),
                    'rate': anomaly.get('anomaly_rate', 0)
                }
                for metric, anomaly in results.get('anomalies', {}).items()
                if isinstance(anomaly, dict)
            }
        }
        
        return {
            'success': True,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for scheduled analysis
async def run_scheduled_analysis():
    """Run scheduled statistical analysis (for cron jobs)."""
    try:
        logger.info("Running scheduled statistical analysis")
        results = analyzer.comprehensive_analysis(days=7)  # Weekly analysis
        
        # Store results or send notifications
        insights = results.get('insights', [])
        if insights:
            high_priority = [i for i in insights if i.get('priority') == 'high']
            if high_priority:
                logger.warning(f"High priority insights detected: {len(high_priority)}")
                # Here you would send notifications or alerts
        
        logger.info(f"Scheduled analysis completed: {results.get('summary', {})}")
        
    except Exception as e:
        logger.error(f"Error in scheduled analysis: {e}")

logger.info("✅ Statistics API router initialized (STORY-006)")