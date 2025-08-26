"""
Statistical Analysis Engine for Security Metrics (STORY-006)
=============================================================

Provides comprehensive statistical analysis capabilities including:
- Trend analysis and regression
- Anomaly detection
- Correlation analysis
- Forecasting and predictions
- Pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
import logging
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Core statistical analysis engine for security metrics."""
    
    def __init__(self, database_path: str = None):
        """Initialize the statistical analyzer."""
        if database_path is None:
            database_path = "backend/cache/gcp_data.db"
        self.database_path = Path(database_path)
        self.scaler = StandardScaler()
        
        # Cache for expensive computations
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("[OK] Statistical analyzer initialized")
    
    def get_metrics_data(self, 
                        metric_type: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch metrics data from database."""
        try:
            # Default to last 30 days if no date range specified
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            with sqlite3.connect(self.database_path) as conn:
                # Map metric types to tables
                table_map = {
                    'security_findings': 'security_findings',
                    'iam_policies': 'iam_policies',
                    'storage_buckets': 'storage_buckets',
                    'firewall_rules': 'assets',
                    'api_keys': 'api_keys',
                    'recommendations': 'recommendations'
                }
                
                table = table_map.get(metric_type, 'assets')
                
                # Fetch data with time filtering if available
                query = f"""
                    SELECT * FROM {table}
                    WHERE datetime(created_at) BETWEEN ? AND ?
                    ORDER BY created_at
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn,
                    params=(start_date.isoformat(), end_date.isoformat()),
                    parse_dates=['created_at']
                )
                
                # If no created_at column, get all data
                if df.empty:
                    query = f"SELECT * FROM {table}"
                    df = pd.read_sql_query(query, conn)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching metrics data: {e}")
            return pd.DataFrame()
    
    def calculate_trends(self, 
                        data: pd.DataFrame,
                        metric_column: str,
                        time_column: str = 'created_at') -> Dict[str, Any]:
        """Calculate trend analysis including regression and moving averages."""
        try:
            if data.empty or metric_column not in data.columns:
                return {'error': 'Invalid data or metric column'}
            
            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(data[metric_column]):
                data[metric_column] = pd.to_numeric(data[metric_column], errors='coerce')
            
            # Remove NaN values
            data = data.dropna(subset=[metric_column])
            
            if len(data) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Linear regression for trend
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[metric_column].values
            
            model = LinearRegression()
            model.fit(X, y)
            trend_line = model.predict(X)
            
            # Calculate slope and direction
            slope = model.coef_[0]
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            
            # Moving averages
            sma_7 = data[metric_column].rolling(window=min(7, len(data))).mean()
            sma_30 = data[metric_column].rolling(window=min(30, len(data))).mean()
            ema = data[metric_column].ewm(span=min(7, len(data)), adjust=False).mean()
            
            # Calculate trend strength (R-squared)
            ss_res = np.sum((y - trend_line) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Identify change points
            if len(data) > 10:
                diff = np.diff(y)
                peaks, _ = find_peaks(np.abs(diff), height=np.std(diff) * 2)
                change_points = peaks.tolist()
            else:
                change_points = []
            
            return {
                'trend_direction': trend_direction,
                'slope': float(slope),
                'r_squared': float(r_squared),
                'trend_strength': 'strong' if abs(r_squared) > 0.7 else 'moderate' if abs(r_squared) > 0.4 else 'weak',
                'current_value': float(y[-1]) if len(y) > 0 else None,
                'mean': float(np.mean(y)),
                'std_dev': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y)),
                'sma_7': sma_7.tolist() if not sma_7.empty else [],
                'sma_30': sma_30.tolist() if not sma_30.empty else [],
                'ema': ema.tolist() if not ema.empty else [],
                'trend_line': trend_line.tolist(),
                'change_points': change_points,
                'data_points': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {'error': str(e)}
    
    def detect_anomalies(self, 
                        data: pd.DataFrame,
                        metric_column: str,
                        sensitivity: float = 2.0) -> Dict[str, Any]:
        """Detect anomalies using multiple methods."""
        try:
            if data.empty or metric_column not in data.columns:
                return {'error': 'Invalid data or metric column'}
            
            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(data[metric_column]):
                data[metric_column] = pd.to_numeric(data[metric_column], errors='coerce')
            
            values = data[metric_column].dropna().values
            
            if len(values) < 3:
                return {'error': 'Insufficient data for anomaly detection'}
            
            anomalies = {}
            
            # Method 1: Z-score based detection
            z_scores = np.abs(stats.zscore(values))
            z_threshold = sensitivity
            z_anomalies = np.where(z_scores > z_threshold)[0]
            
            # Method 2: IQR based detection
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - sensitivity * IQR
            upper_bound = Q3 + sensitivity * IQR
            iqr_anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
            
            # Method 3: Isolation Forest (if enough data)
            if len(values) > 20:
                iso_forest = IsolationForest(
                    contamination=min(0.1, 5/len(values)),
                    random_state=42
                )
                predictions = iso_forest.fit_predict(values.reshape(-1, 1))
                iso_anomalies = np.where(predictions == -1)[0]
            else:
                iso_anomalies = np.array([])
            
            # Combine anomalies (intersection for higher confidence)
            all_anomalies = np.unique(np.concatenate([z_anomalies, iqr_anomalies]))
            high_confidence = np.intersect1d(z_anomalies, iqr_anomalies)
            
            # Calculate anomaly score for each point
            anomaly_scores = z_scores.tolist()
            
            return {
                'total_anomalies': len(all_anomalies),
                'high_confidence_anomalies': len(high_confidence),
                'anomaly_indices': all_anomalies.tolist(),
                'high_confidence_indices': high_confidence.tolist(),
                'anomaly_rate': float(len(all_anomalies) / len(values)) if len(values) > 0 else 0,
                'methods': {
                    'z_score': {
                        'count': len(z_anomalies),
                        'indices': z_anomalies.tolist(),
                        'threshold': z_threshold
                    },
                    'iqr': {
                        'count': len(iqr_anomalies),
                        'indices': iqr_anomalies.tolist(),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    },
                    'isolation_forest': {
                        'count': len(iso_anomalies),
                        'indices': iso_anomalies.tolist()
                    } if len(iso_anomalies) > 0 else None
                },
                'anomaly_scores': anomaly_scores,
                'statistics': {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std_dev': float(np.std(values)),
                    'q1': float(Q1),
                    'q3': float(Q3),
                    'iqr': float(IQR)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'error': str(e)}
    
    def correlation_analysis(self, 
                            data: pd.DataFrame,
                            metrics: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis between multiple metrics."""
        try:
            if data.empty or not metrics:
                return {'error': 'Invalid data or metrics'}
            
            # Filter to only specified metrics
            available_metrics = [m for m in metrics if m in data.columns]
            
            if len(available_metrics) < 2:
                return {'error': 'Need at least 2 metrics for correlation'}
            
            # Ensure numeric data
            for metric in available_metrics:
                if not pd.api.types.is_numeric_dtype(data[metric]):
                    data[metric] = pd.to_numeric(data[metric], errors='coerce')
            
            # Calculate correlations
            correlation_data = data[available_metrics].dropna()
            
            if correlation_data.empty:
                return {'error': 'No valid data for correlation'}
            
            # Pearson correlation
            pearson_corr = correlation_data.corr(method='pearson')
            
            # Spearman correlation (rank-based)
            spearman_corr = correlation_data.corr(method='spearman')
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(available_metrics)):
                for j in range(i+1, len(available_metrics)):
                    metric1 = available_metrics[i]
                    metric2 = available_metrics[j]
                    pearson_val = pearson_corr.loc[metric1, metric2]
                    spearman_val = spearman_corr.loc[metric1, metric2]
                    
                    if abs(pearson_val) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'pearson': float(pearson_val),
                            'spearman': float(spearman_val),
                            'strength': 'strong' if abs(pearson_val) > 0.7 else 'moderate'
                        })
            
            # Sort by absolute correlation strength
            strong_correlations.sort(key=lambda x: abs(x['pearson']), reverse=True)
            
            return {
                'pearson_correlation': pearson_corr.to_dict(),
                'spearman_correlation': spearman_corr.to_dict(),
                'strong_correlations': strong_correlations,
                'metrics_analyzed': available_metrics,
                'sample_size': len(correlation_data),
                'insights': self._generate_correlation_insights(strong_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e)}
    
    def forecast(self, 
                data: pd.DataFrame,
                metric_column: str,
                horizon: int = 7) -> Dict[str, Any]:
        """Simple forecasting using linear regression and moving averages."""
        try:
            if data.empty or metric_column not in data.columns:
                return {'error': 'Invalid data or metric column'}
            
            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(data[metric_column]):
                data[metric_column] = pd.to_numeric(data[metric_column], errors='coerce')
            
            values = data[metric_column].dropna().values
            
            if len(values) < 3:
                return {'error': 'Insufficient data for forecasting'}
            
            # Linear regression forecast
            X = np.arange(len(values)).reshape(-1, 1)
            y = values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future values
            future_X = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            # Calculate confidence intervals (simplified)
            std_error = np.std(y - model.predict(X))
            confidence_lower = forecast_values - 1.96 * std_error
            confidence_upper = forecast_values + 1.96 * std_error
            
            # Moving average forecast (simple method)
            ma_window = min(7, len(values) // 2)
            ma_forecast = [np.mean(values[-ma_window:])] * horizon
            
            # Calculate forecast accuracy metrics (on historical data)
            if len(values) > 10:
                train_size = int(len(values) * 0.8)
                train_X = X[:train_size]
                train_y = y[:train_size]
                test_X = X[train_size:]
                test_y = y[train_size:]
                
                model.fit(train_X, train_y)
                predictions = model.predict(test_X)
                
                mape = np.mean(np.abs((test_y - predictions) / test_y)) * 100 if np.all(test_y != 0) else 0
                rmse = np.sqrt(np.mean((test_y - predictions) ** 2))
            else:
                mape = None
                rmse = None
            
            return {
                'forecast_values': forecast_values.tolist(),
                'confidence_lower': confidence_lower.tolist(),
                'confidence_upper': confidence_upper.tolist(),
                'ma_forecast': ma_forecast,
                'horizon_days': horizon,
                'method': 'linear_regression',
                'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                'accuracy_metrics': {
                    'mape': float(mape) if mape is not None else None,
                    'rmse': float(rmse) if rmse is not None else None,
                    'confidence_level': 0.95
                },
                'historical_mean': float(np.mean(values)),
                'historical_std': float(np.std(values)),
                'last_value': float(values[-1]) if len(values) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return {'error': str(e)}
    
    def pattern_recognition(self, 
                          data: pd.DataFrame,
                          metric_column: str) -> Dict[str, Any]:
        """Identify patterns in the data."""
        try:
            if data.empty or metric_column not in data.columns:
                return {'error': 'Invalid data or metric column'}
            
            patterns = {
                'seasonality': None,
                'recurring_patterns': [],
                'trend_changes': [],
                'volatility_periods': []
            }
            
            # Ensure numeric data
            if not pd.api.types.is_numeric_dtype(data[metric_column]):
                data[metric_column] = pd.to_numeric(data[metric_column], errors='coerce')
            
            values = data[metric_column].dropna().values
            
            if len(values) < 7:
                return patterns
            
            # Check for weekly patterns (if enough data)
            if len(values) >= 14:
                weekly_avg = []
                for i in range(7):
                    day_values = values[i::7]
                    if len(day_values) > 0:
                        weekly_avg.append(np.mean(day_values))
                
                if weekly_avg:
                    patterns['seasonality'] = {
                        'type': 'weekly',
                        'pattern': weekly_avg,
                        'peak_day': int(np.argmax(weekly_avg)),
                        'low_day': int(np.argmin(weekly_avg))
                    }
            
            # Detect volatility periods
            if len(values) > 10:
                window = min(7, len(values) // 3)
                rolling_std = pd.Series(values).rolling(window=window).std()
                high_volatility_threshold = np.percentile(rolling_std.dropna(), 75)
                
                high_vol_periods = np.where(rolling_std > high_volatility_threshold)[0]
                if len(high_vol_periods) > 0:
                    patterns['volatility_periods'] = high_vol_periods.tolist()
            
            # Detect trend changes
            if len(values) > 20:
                window = 5
                slopes = []
                for i in range(window, len(values) - window):
                    x = np.arange(window)
                    y = values[i-window:i]
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                
                if slopes:
                    slope_changes = np.diff(np.sign(slopes))
                    change_points = np.where(slope_changes != 0)[0]
                    patterns['trend_changes'] = change_points.tolist()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            return {'error': str(e)}
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from analysis results."""
        insights = []
        
        try:
            # Trend insights
            if 'trends' in analysis_results:
                for metric, trend in analysis_results['trends'].items():
                    if 'trend_direction' in trend:
                        if trend['trend_direction'] == 'increasing' and trend.get('slope', 0) > 0.1:
                            insights.append({
                                'type': 'trend',
                                'priority': 'high' if trend.get('r_squared', 0) > 0.7 else 'medium',
                                'metric': metric,
                                'insight': f"{metric} shows strong upward trend ({trend['slope']:.2f} per day)",
                                'recommendation': f"Investigate root cause of increasing {metric}",
                                'confidence': trend.get('r_squared', 0)
                            })
            
            # Anomaly insights
            if 'anomalies' in analysis_results:
                for metric, anomaly in analysis_results['anomalies'].items():
                    if 'total_anomalies' in anomaly and anomaly['total_anomalies'] > 0:
                        insights.append({
                            'type': 'anomaly',
                            'priority': 'high' if anomaly.get('high_confidence_anomalies', 0) > 2 else 'medium',
                            'metric': metric,
                            'insight': f"Detected {anomaly['total_anomalies']} anomalies in {metric}",
                            'recommendation': "Review anomalous events for security implications",
                            'confidence': anomaly.get('anomaly_rate', 0)
                        })
            
            # Correlation insights
            if 'correlations' in analysis_results:
                corr = analysis_results['correlations']
                if 'strong_correlations' in corr:
                    for correlation in corr['strong_correlations'][:3]:  # Top 3
                        insights.append({
                            'type': 'correlation',
                            'priority': 'medium',
                            'metrics': [correlation['metric1'], correlation['metric2']],
                            'insight': f"Strong correlation ({correlation['pearson']:.2f}) between {correlation['metric1']} and {correlation['metric2']}",
                            'recommendation': "Consider combined monitoring and alerting",
                            'confidence': abs(correlation['pearson'])
                        })
            
            # Forecast insights
            if 'forecasts' in analysis_results:
                for metric, forecast in analysis_results['forecasts'].items():
                    if 'trend_direction' in forecast:
                        last_val = forecast.get('last_value', 0)
                        forecast_val = forecast.get('forecast_values', [0])[0] if forecast.get('forecast_values') else 0
                        change_pct = ((forecast_val - last_val) / last_val * 100) if last_val != 0 else 0
                        
                        if abs(change_pct) > 20:
                            insights.append({
                                'type': 'forecast',
                                'priority': 'high' if abs(change_pct) > 50 else 'medium',
                                'metric': metric,
                                'insight': f"{metric} predicted to change by {change_pct:.1f}% in next week",
                                'recommendation': f"Prepare for {'increase' if change_pct > 0 else 'decrease'} in {metric}",
                                'confidence': 1 - (forecast.get('accuracy_metrics', {}).get('mape', 100) / 100)
                            })
            
            # Pattern insights
            if 'patterns' in analysis_results:
                for metric, pattern in analysis_results['patterns'].items():
                    if pattern.get('seasonality'):
                        insights.append({
                            'type': 'pattern',
                            'priority': 'low',
                            'metric': metric,
                            'insight': f"{metric} shows weekly seasonality pattern",
                            'recommendation': "Adjust monitoring thresholds based on day of week",
                            'confidence': 0.7
                        })
            
            # Sort by priority and confidence
            insights.sort(key=lambda x: (
                {'high': 0, 'medium': 1, 'low': 2}[x['priority']],
                -x['confidence']
            ))
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    def _generate_correlation_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate text insights from correlations."""
        insights = []
        for corr in correlations[:3]:  # Top 3
            if corr['pearson'] > 0.7:
                insights.append(f"Strong positive correlation between {corr['metric1']} and {corr['metric2']}")
            elif corr['pearson'] < -0.7:
                insights.append(f"Strong negative correlation between {corr['metric1']} and {corr['metric2']}")
        return insights
    
    def comprehensive_analysis(self, 
                             metric_types: List[str] = None,
                             days: int = 30) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on all metrics."""
        try:
            if metric_types is None:
                metric_types = ['security_findings', 'iam_policies', 'storage_buckets']
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'period_days': days,
                'metrics_analyzed': metric_types,
                'trends': {},
                'anomalies': {},
                'correlations': {},
                'forecasts': {},
                'patterns': {},
                'insights': []
            }
            
            # Analyze each metric type
            for metric_type in metric_types:
                data = self.get_metrics_data(metric_type)
                
                if not data.empty:
                    # Find numeric columns for analysis
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        # Use first numeric column or specific column based on metric type
                        metric_col = self._get_primary_metric_column(metric_type, numeric_cols)
                        
                        if metric_col:
                            # Trend analysis
                            results['trends'][metric_type] = self.calculate_trends(data, metric_col)
                            
                            # Anomaly detection
                            results['anomalies'][metric_type] = self.detect_anomalies(data, metric_col)
                            
                            # Forecasting
                            results['forecasts'][metric_type] = self.forecast(data, metric_col)
                            
                            # Pattern recognition
                            results['patterns'][metric_type] = self.pattern_recognition(data, metric_col)
            
            # Cross-metric correlation (if we have multiple metrics)
            if len(results['trends']) > 1:
                # Create combined dataset for correlation
                combined_data = pd.DataFrame()
                for metric_type in metric_types[:3]:  # Limit to 3 for performance
                    data = self.get_metrics_data(metric_type)
                    if not data.empty:
                        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            metric_col = self._get_primary_metric_column(metric_type, numeric_cols)
                            if metric_col and metric_col in data.columns:
                                combined_data[metric_type] = data[metric_col].values[:100]  # Limit rows
                
                if not combined_data.empty and len(combined_data.columns) > 1:
                    results['correlations'] = self.correlation_analysis(
                        combined_data, 
                        combined_data.columns.tolist()
                    )
            
            # Generate insights
            results['insights'] = self.generate_insights(results)
            
            # Summary statistics
            results['summary'] = {
                'total_metrics_analyzed': len(metric_types),
                'total_anomalies_detected': sum(
                    a.get('total_anomalies', 0) 
                    for a in results['anomalies'].values() 
                    if isinstance(a, dict)
                ),
                'strong_correlations_found': len(
                    results.get('correlations', {}).get('strong_correlations', [])
                ),
                'insights_generated': len(results['insights']),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[OK] Comprehensive analysis completed: {results['summary']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e)}
    
    def _get_primary_metric_column(self, metric_type: str, numeric_cols: List[str]) -> Optional[str]:
        """Get the primary metric column for a given metric type."""
        # Map metric types to their primary numeric columns
        primary_columns = {
            'security_findings': 'severity_score',
            'iam_policies': 'member_count',
            'storage_buckets': 'size_bytes',
            'firewall_rules': 'priority',
            'api_keys': 'usage_count',
            'recommendations': 'priority_score'
        }
        
        preferred = primary_columns.get(metric_type)
        if preferred and preferred in numeric_cols:
            return preferred
        
        # Fallback to first numeric column
        return numeric_cols[0] if numeric_cols else None