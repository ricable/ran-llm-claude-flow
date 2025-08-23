#!/usr/bin/env python3
"""
Business Intelligence Engine for Production Analytics
Phase 4 Advanced Analytics - ML-Powered BI Reports

Features:
- Cost analysis and ROI calculations
- System utilization optimization reports
- Resource planning and capacity forecasting
- Executive dashboards and KPI tracking
- Automated business insights generation
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BusinessMetric:
    """Business metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    cost: float
    department: str
    project: str
    tags: Dict[str, str]

@dataclass
class ROIAnalysis:
    """Return on Investment analysis result"""
    investment: float
    return_value: float
    roi_percentage: float
    payback_period_days: int
    net_present_value: float
    confidence_score: float

@dataclass
class UtilizationReport:
    """System utilization analysis report"""
    resource_type: str
    current_utilization: float
    peak_utilization: float
    avg_utilization: float
    efficiency_score: float
    optimization_potential: float
    recommendations: List[str]

@dataclass
class CostBreakdown:
    """Cost analysis breakdown"""
    total_cost: float
    by_department: Dict[str, float]
    by_project: Dict[str, float]
    by_resource_type: Dict[str, float]
    trends: Dict[str, float]  # percentage changes
    forecasted_cost: float

class BusinessIntelligenceEngine:
    """Advanced business intelligence engine with ML insights"""
    
    def __init__(self, db_path: str = "bi_analytics.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.kmeans_model = KMeans(n_clusters=5, random_state=42)
        
        # Business configuration
        self.cost_rates = {
            'compute': 0.12,  # per hour
            'storage': 0.023, # per GB
            'network': 0.09,  # per GB
            'labor': 75.0,    # per hour
        }
        
        self.discount_rate = 0.08  # 8% for NPV calculations
        self.target_utilization = {
            'cpu': 0.75,
            'memory': 0.85,
            'storage': 0.90,
            'network': 0.70
        }
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for analytics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create business metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                value REAL,
                cost REAL,
                department TEXT,
                project TEXT,
                tags TEXT
            )
        ''')
        
        # Create ROI analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS roi_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                investment REAL,
                return_value REAL,
                roi_percentage REAL,
                payback_period_days INTEGER,
                net_present_value REAL,
                confidence_score REAL
            )
        ''')
        
        # Create utilization reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS utilization_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                resource_type TEXT,
                current_utilization REAL,
                peak_utilization REAL,
                avg_utilization REAL,
                efficiency_score REAL,
                optimization_potential REAL,
                recommendations TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_business_metric(self, metric: BusinessMetric):
        """Add business metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO business_metrics 
            (timestamp, metric_name, value, cost, department, project, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp.isoformat(),
            metric.metric_name,
            metric.value,
            metric.cost,
            metric.department,
            metric.project,
            json.dumps(metric.tags)
        ))
        
        conn.commit()
        conn.close()
        
    def calculate_roi_analysis(self, 
                             investment: float, 
                             revenue_stream: List[float],
                             time_periods: int = 12) -> ROIAnalysis:
        """Calculate comprehensive ROI analysis"""
        
        # Calculate total return value
        total_return = sum(revenue_stream)
        
        # Basic ROI calculation
        roi_percentage = ((total_return - investment) / investment) * 100
        
        # Calculate payback period
        cumulative_return = 0
        payback_period_days = 0
        for i, monthly_return in enumerate(revenue_stream):
            cumulative_return += monthly_return
            if cumulative_return >= investment:
                payback_period_days = (i + 1) * 30  # assuming monthly periods
                break
        else:
            payback_period_days = len(revenue_stream) * 30
            
        # Calculate Net Present Value (NPV)
        monthly_discount_rate = self.discount_rate / 12
        npv = -investment
        for i, cash_flow in enumerate(revenue_stream):
            npv += cash_flow / ((1 + monthly_discount_rate) ** (i + 1))
            
        # Confidence score based on data quality and variability
        revenue_std = np.std(revenue_stream) if len(revenue_stream) > 1 else 0
        revenue_mean = np.mean(revenue_stream) if revenue_stream else 0
        cv = revenue_std / revenue_mean if revenue_mean > 0 else 1
        confidence_score = max(0.1, min(1.0, 1.0 - cv))  # Higher variability = lower confidence
        
        roi_analysis = ROIAnalysis(
            investment=investment,
            return_value=total_return,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period_days,
            net_present_value=npv,
            confidence_score=confidence_score
        )
        
        # Store in database
        self._store_roi_analysis(roi_analysis)
        
        return roi_analysis
        
    def analyze_system_utilization(self, 
                                 metrics_data: List[BusinessMetric],
                                 resource_type: str) -> UtilizationReport:
        """Analyze system resource utilization and efficiency"""
        
        # Filter metrics for specific resource type
        resource_metrics = [m for m in metrics_data if resource_type in m.metric_name.lower()]
        
        if not resource_metrics:
            logger.warning(f"No metrics found for resource type: {resource_type}")
            return UtilizationReport(
                resource_type=resource_type,
                current_utilization=0.0,
                peak_utilization=0.0,
                avg_utilization=0.0,
                efficiency_score=0.0,
                optimization_potential=0.0,
                recommendations=["No data available for analysis"]
            )
            
        values = [m.value for m in resource_metrics]
        
        # Calculate utilization statistics
        current_utilization = values[-1] if values else 0.0
        peak_utilization = max(values) if values else 0.0
        avg_utilization = np.mean(values) if values else 0.0
        
        # Calculate efficiency score (how close to target utilization)
        target = self.target_utilization.get(resource_type, 0.75)
        efficiency_score = 1.0 - abs(avg_utilization - target) / target
        efficiency_score = max(0.0, min(1.0, efficiency_score))
        
        # Calculate optimization potential
        waste = max(0, avg_utilization - target) if avg_utilization > target else 0
        underutilization = max(0, target - avg_utilization) if avg_utilization < target else 0
        optimization_potential = (waste + underutilization) * 100  # as percentage
        
        # Generate recommendations
        recommendations = self._generate_utilization_recommendations(
            resource_type, current_utilization, avg_utilization, target
        )
        
        report = UtilizationReport(
            resource_type=resource_type,
            current_utilization=current_utilization,
            peak_utilization=peak_utilization,
            avg_utilization=avg_utilization,
            efficiency_score=efficiency_score,
            optimization_potential=optimization_potential,
            recommendations=recommendations
        )
        
        # Store in database
        self._store_utilization_report(report)
        
        return report
        
    def generate_cost_breakdown(self, 
                              metrics_data: List[BusinessMetric],
                              time_period: int = 30) -> CostBreakdown:
        """Generate comprehensive cost breakdown and analysis"""
        
        # Filter data for specified time period (days)
        cutoff_date = datetime.now() - timedelta(days=time_period)
        recent_metrics = [m for m in metrics_data if m.timestamp >= cutoff_date]
        
        if not recent_metrics:
            return CostBreakdown(
                total_cost=0.0,
                by_department={},
                by_project={},
                by_resource_type={},
                trends={},
                forecasted_cost=0.0
            )
            
        # Calculate total cost
        total_cost = sum(m.cost for m in recent_metrics)
        
        # Breakdown by department
        by_department = defaultdict(float)
        for metric in recent_metrics:
            by_department[metric.department] += metric.cost
            
        # Breakdown by project
        by_project = defaultdict(float)
        for metric in recent_metrics:
            by_project[metric.project] += metric.cost
            
        # Breakdown by resource type
        by_resource_type = defaultdict(float)
        for metric in recent_metrics:
            resource_type = self._classify_resource_type(metric.metric_name)
            by_resource_type[resource_type] += metric.cost
            
        # Calculate trends (compare with previous period)
        previous_period_start = cutoff_date - timedelta(days=time_period)
        previous_metrics = [m for m in metrics_data 
                          if previous_period_start <= m.timestamp < cutoff_date]
        
        trends = {}
        if previous_metrics:
            prev_total = sum(m.cost for m in previous_metrics)
            trends['total'] = ((total_cost - prev_total) / prev_total * 100) if prev_total > 0 else 0
            
            # Department trends
            prev_by_dept = defaultdict(float)
            for metric in previous_metrics:
                prev_by_dept[metric.department] += metric.cost
                
            for dept in by_department:
                if dept in prev_by_dept:
                    change = ((by_department[dept] - prev_by_dept[dept]) / prev_by_dept[dept] * 100)
                    trends[f"dept_{dept}"] = change
                    
        # Forecast next period cost using linear regression
        forecasted_cost = self._forecast_cost(recent_metrics, time_period)
        
        return CostBreakdown(
            total_cost=total_cost,
            by_department=dict(by_department),
            by_project=dict(by_project),
            by_resource_type=dict(by_resource_type),
            trends=trends,
            forecasted_cost=forecasted_cost
        )
        
    def generate_executive_dashboard(self, metrics_data: List[BusinessMetric]) -> Dict[str, Any]:
        """Generate executive-level dashboard with KPIs and insights"""
        
        # Key Performance Indicators
        kpis = self._calculate_kpis(metrics_data)
        
        # Cost efficiency metrics
        cost_breakdown = self.generate_cost_breakdown(metrics_data)
        
        # Resource utilization summary
        utilization_summary = self._get_utilization_summary(metrics_data)
        
        # Business insights using ML
        insights = self._generate_ml_insights(metrics_data)
        
        # Performance trends
        trends = self._analyze_performance_trends(metrics_data)
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'kpis': kpis,
            'cost_analysis': {
                'total_monthly_cost': cost_breakdown.total_cost,
                'cost_trend': cost_breakdown.trends.get('total', 0),
                'top_cost_centers': dict(sorted(cost_breakdown.by_department.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]),
                'forecasted_cost': cost_breakdown.forecasted_cost
            },
            'resource_efficiency': {
                'overall_efficiency': utilization_summary['avg_efficiency'],
                'optimization_opportunity': utilization_summary['total_optimization'],
                'critical_resources': utilization_summary['critical_resources']
            },
            'business_insights': insights,
            'performance_trends': trends,
            'recommendations': self._generate_executive_recommendations(metrics_data)
        }
        
        return dashboard
        
    def create_utilization_heatmap(self, metrics_data: List[BusinessMetric]) -> go.Figure:
        """Create utilization heatmap visualization"""
        
        # Prepare data for heatmap
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'resource': self._classify_resource_type(m.metric_name),
                'utilization': m.value,
                'department': m.department
            }
            for m in metrics_data
        ])
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Group by hour and resource
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        pivot_data = df.pivot_table(
            values='utilization',
            index='resource',
            columns='hour',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=list(range(24)),
            y=pivot_data.index,
            colorscale='RdYlGn_r',
            hoverongap=False,
            hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Utilization: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Resource Utilization Heatmap (24-Hour View)',
            xaxis_title='Hour of Day',
            yaxis_title='Resource Type',
            height=500
        )
        
        return fig
        
    def create_cost_trend_chart(self, metrics_data: List[BusinessMetric]) -> go.Figure:
        """Create cost trend analysis chart"""
        
        df = pd.DataFrame([
            {
                'date': m.timestamp.date(),
                'cost': m.cost,
                'department': m.department,
                'project': m.project
            }
            for m in metrics_data
        ])
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No cost data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
            
        # Daily cost aggregation
        daily_costs = df.groupby('date')['cost'].sum().reset_index()
        
        # Department breakdown
        dept_costs = df.groupby(['date', 'department'])['cost'].sum().reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Daily Costs', 'Cost by Department'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Total cost trend
        fig.add_trace(
            go.Scatter(
                x=daily_costs['date'],
                y=daily_costs['cost'],
                mode='lines+markers',
                name='Total Cost',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # Department costs
        departments = dept_costs['department'].unique()
        colors = px.colors.qualitative.Set3[:len(departments)]
        
        for dept, color in zip(departments, colors):
            dept_data = dept_costs[dept_costs['department'] == dept]
            fig.add_trace(
                go.Scatter(
                    x=dept_data['date'],
                    y=dept_data['cost'],
                    mode='lines',
                    name=dept,
                    line=dict(color=color),
                    stackgroup='one'
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title='Cost Analysis Trends',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=2, col=1)
        
        return fig
        
    def _calculate_kpis(self, metrics_data: List[BusinessMetric]) -> Dict[str, float]:
        """Calculate key performance indicators"""
        if not metrics_data:
            return {}
            
        # Calculate various KPIs
        total_cost = sum(m.cost for m in metrics_data)
        avg_utilization = np.mean([m.value for m in metrics_data])
        
        # Efficiency metrics
        target_costs = self._calculate_target_costs(metrics_data)
        cost_efficiency = (target_costs / total_cost * 100) if total_cost > 0 else 0
        
        # Performance metrics
        performance_score = self._calculate_performance_score(metrics_data)
        
        return {
            'total_cost': total_cost,
            'cost_efficiency': cost_efficiency,
            'avg_utilization': avg_utilization,
            'performance_score': performance_score,
            'cost_per_transaction': total_cost / len(metrics_data) if metrics_data else 0
        }
        
    def _get_utilization_summary(self, metrics_data: List[BusinessMetric]) -> Dict[str, Any]:
        """Get summary of resource utilization across all types"""
        resource_types = ['cpu', 'memory', 'storage', 'network']
        utilization_reports = []
        
        for resource_type in resource_types:
            report = self.analyze_system_utilization(metrics_data, resource_type)
            utilization_reports.append(report)
            
        avg_efficiency = np.mean([r.efficiency_score for r in utilization_reports])
        total_optimization = sum([r.optimization_potential for r in utilization_reports])
        
        # Find critical resources (low efficiency or high optimization potential)
        critical_resources = [
            r.resource_type for r in utilization_reports
            if r.efficiency_score < 0.6 or r.optimization_potential > 20
        ]
        
        return {
            'avg_efficiency': avg_efficiency,
            'total_optimization': total_optimization,
            'critical_resources': critical_resources,
            'reports': utilization_reports
        }
        
    def _generate_ml_insights(self, metrics_data: List[BusinessMetric]) -> List[Dict[str, Any]]:
        """Generate ML-powered business insights"""
        insights = []
        
        if len(metrics_data) < 10:
            return insights
            
        # Prepare data for ML analysis
        df = pd.DataFrame([
            {
                'value': m.value,
                'cost': m.cost,
                'hour': m.timestamp.hour,
                'day_of_week': m.timestamp.weekday(),
                'department_hash': hash(m.department) % 1000,
                'project_hash': hash(m.project) % 1000
            }
            for m in metrics_data
        ])
        
        # Cluster analysis for cost optimization
        try:
            features = df[['value', 'cost', 'hour', 'day_of_week']].values
            features_scaled = self.scaler.fit_transform(features)
            clusters = self.kmeans_model.fit_predict(features_scaled)
            
            df['cluster'] = clusters
            
            # Analyze clusters for insights
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_data = df[df['cluster'] == cluster_id]
                if len(cluster_data) > 5:
                    insight = self._analyze_cluster(cluster_id, cluster_data)
                    if insight:
                        insights.append(insight)
                        
        except Exception as e:
            logger.error(f"ML insights generation error: {e}")
            
        # Cost anomaly detection
        cost_anomalies = self._detect_cost_anomalies(metrics_data)
        if cost_anomalies:
            insights.append({
                'type': 'cost_anomaly',
                'description': f"Detected {len(cost_anomalies)} cost anomalies",
                'severity': 'high' if len(cost_anomalies) > 5 else 'medium',
                'details': cost_anomalies[:3]  # Top 3 anomalies
            })
            
        return insights
        
    def _analyze_performance_trends(self, metrics_data: List[BusinessMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(metrics_data) < 7:
            return {}
            
        # Sort by timestamp
        sorted_metrics = sorted(metrics_data, key=lambda x: x.timestamp)
        
        # Calculate daily averages
        daily_data = defaultdict(list)
        for metric in sorted_metrics:
            date_key = metric.timestamp.date()
            daily_data[date_key].append(metric.value)
            
        daily_averages = {
            date: np.mean(values)
            for date, values in daily_data.items()
        }
        
        # Calculate trends
        dates = sorted(daily_averages.keys())
        values = [daily_averages[date] for date in dates]
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        trend_strength = abs(slope)
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'daily_averages': dict(daily_averages),
            'volatility': np.std(values),
            'growth_rate': (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        }
        
    def _generate_executive_recommendations(self, metrics_data: List[BusinessMetric]) -> List[Dict[str, str]]:
        """Generate executive-level recommendations"""
        recommendations = []
        
        # Cost optimization recommendations
        cost_breakdown = self.generate_cost_breakdown(metrics_data)
        if cost_breakdown.trends.get('total', 0) > 15:  # More than 15% cost increase
            recommendations.append({
                'category': 'cost_optimization',
                'priority': 'high',
                'title': 'Rising Cost Trend Detected',
                'description': f"Costs have increased by {cost_breakdown.trends.get('total', 0):.1f}% this period",
                'action': 'Implement cost control measures and review high-spend departments'
            })
            
        # Utilization optimization
        utilization_summary = self._get_utilization_summary(metrics_data)
        if utilization_summary['avg_efficiency'] < 0.7:
            recommendations.append({
                'category': 'efficiency',
                'priority': 'medium',
                'title': 'Resource Efficiency Below Target',
                'description': f"Average efficiency is {utilization_summary['avg_efficiency']:.1%}",
                'action': 'Optimize resource allocation and consider scaling adjustments'
            })
            
        # Performance recommendations
        performance_trends = self._analyze_performance_trends(metrics_data)
        if performance_trends.get('trend_direction') == 'decreasing':
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'title': 'Performance Degradation Detected',
                'description': 'System performance is trending downward',
                'action': 'Investigate performance bottlenecks and implement improvements'
            })
            
        return recommendations
        
    def _classify_resource_type(self, metric_name: str) -> str:
        """Classify metric into resource type"""
        metric_lower = metric_name.lower()
        if 'cpu' in metric_lower or 'processor' in metric_lower:
            return 'cpu'
        elif 'memory' in metric_lower or 'ram' in metric_lower:
            return 'memory'
        elif 'disk' in metric_lower or 'storage' in metric_lower:
            return 'storage'
        elif 'network' in metric_lower or 'bandwidth' in metric_lower:
            return 'network'
        else:
            return 'other'
            
    def _generate_utilization_recommendations(self, 
                                           resource_type: str, 
                                           current: float, 
                                           average: float, 
                                           target: float) -> List[str]:
        """Generate utilization optimization recommendations"""
        recommendations = []
        
        if current > 0.9:
            recommendations.append(f"CRITICAL: {resource_type} utilization at {current:.1%} - immediate scaling required")
        elif current > 0.85:
            recommendations.append(f"HIGH: {resource_type} utilization at {current:.1%} - consider scaling soon")
            
        if average < target * 0.5:
            recommendations.append(f"Underutilized: {resource_type} averaging {average:.1%} - consider downsizing")
        elif average > target:
            recommendations.append(f"Over-target: {resource_type} averaging {average:.1%} - optimize or scale")
            
        if not recommendations:
            recommendations.append(f"{resource_type} utilization is within optimal range")
            
        return recommendations
        
    def _forecast_cost(self, metrics_data: List[BusinessMetric], days: int) -> float:
        """Forecast future costs using linear regression"""
        if len(metrics_data) < 7:
            return sum(m.cost for m in metrics_data)
            
        # Group by date and sum costs
        daily_costs = defaultdict(float)
        for metric in metrics_data:
            date_key = metric.timestamp.date()
            daily_costs[date_key] += metric.cost
            
        # Linear regression on daily costs
        dates = sorted(daily_costs.keys())
        costs = [daily_costs[date] for date in dates]
        
        if len(costs) > 1:
            x = np.arange(len(costs))
            slope, intercept = np.polyfit(x, costs, 1)
            
            # Forecast for next period
            future_days = len(costs) + days
            forecasted_daily = slope * future_days + intercept
            return max(0, forecasted_daily * days)  # Ensure non-negative
        else:
            return costs[0] * days if costs else 0.0
            
    def _calculate_target_costs(self, metrics_data: List[BusinessMetric]) -> float:
        """Calculate what costs should be at optimal efficiency"""
        total_target = 0.0
        
        for metric in metrics_data:
            resource_type = self._classify_resource_type(metric.metric_name)
            target_util = self.target_utilization.get(resource_type, 0.75)
            
            # Calculate what cost would be at target utilization
            if metric.value > 0:
                efficiency_ratio = target_util / metric.value
                target_cost = metric.cost * efficiency_ratio
                total_target += target_cost
            else:
                total_target += metric.cost
                
        return total_target
        
    def _calculate_performance_score(self, metrics_data: List[BusinessMetric]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics_data:
            return 0.0
            
        # Factors: utilization efficiency, cost efficiency, stability
        utilizations = [m.value for m in metrics_data]
        costs = [m.cost for m in metrics_data]
        
        # Utilization score (closer to target is better)
        util_scores = []
        for metric in metrics_data:
            resource_type = self._classify_resource_type(metric.metric_name)
            target = self.target_utilization.get(resource_type, 0.75)
            util_score = 1.0 - abs(metric.value - target) / target
            util_scores.append(max(0.0, util_score))
            
        # Cost efficiency (lower cost per unit of utilization is better)
        cost_efficiency_scores = []
        for metric in metrics_data:
            if metric.value > 0:
                cost_per_util = metric.cost / metric.value
                # Normalize against median cost per util
                median_cost_per_util = np.median([m.cost / m.value for m in metrics_data if m.value > 0])
                efficiency = median_cost_per_util / cost_per_util if cost_per_util > 0 else 1.0
                cost_efficiency_scores.append(min(1.0, efficiency))
            else:
                cost_efficiency_scores.append(0.5)  # Neutral score for zero utilization
                
        # Stability score (lower variance is better)
        stability_score = 1.0 / (1.0 + np.std(utilizations)) if len(utilizations) > 1 else 1.0
        
        # Weighted average
        performance_score = (
            np.mean(util_scores) * 0.4 +
            np.mean(cost_efficiency_scores) * 0.4 +
            stability_score * 0.2
        ) * 100
        
        return max(0.0, min(100.0, performance_score))
        
    def _analyze_cluster(self, cluster_id: int, cluster_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze a data cluster for business insights"""
        if len(cluster_data) < 3:
            return None
            
        avg_cost = cluster_data['cost'].mean()
        avg_value = cluster_data['value'].mean()
        
        # Cost efficiency of this cluster
        cost_per_value = avg_cost / avg_value if avg_value > 0 else float('inf')
        
        # Time patterns
        common_hours = cluster_data['hour'].mode().tolist()
        common_days = cluster_data['day_of_week'].mode().tolist()
        
        return {
            'type': 'cluster_analysis',
            'cluster_id': cluster_id,
            'description': f"Cluster {cluster_id}: {len(cluster_data)} data points",
            'avg_cost': avg_cost,
            'avg_utilization': avg_value,
            'cost_efficiency': cost_per_value,
            'patterns': {
                'common_hours': common_hours,
                'common_days': common_days
            },
            'recommendation': self._get_cluster_recommendation(cost_per_value, avg_value)
        }
        
    def _get_cluster_recommendation(self, cost_per_value: float, avg_value: float) -> str:
        """Get recommendation for a cluster based on its characteristics"""
        if cost_per_value == float('inf'):
            return "Investigate zero-utilization high-cost resources"
        elif cost_per_value > 2.0:  # High cost per unit
            return "High-cost cluster - investigate optimization opportunities"
        elif avg_value > 0.9:  # High utilization
            return "High-utilization cluster - monitor for capacity needs"
        elif avg_value < 0.3:  # Low utilization
            return "Low-utilization cluster - consider resource consolidation"
        else:
            return "Well-balanced cluster - maintain current configuration"
            
    def _detect_cost_anomalies(self, metrics_data: List[BusinessMetric]) -> List[Dict[str, Any]]:
        """Detect cost anomalies using statistical methods"""
        costs = [m.cost for m in metrics_data]
        
        if len(costs) < 10:
            return []
            
        # Calculate z-scores for cost anomaly detection
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        anomalies = []
        for i, metric in enumerate(metrics_data):
            if std_cost > 0:
                z_score = abs((metric.cost - mean_cost) / std_cost)
                if z_score > 2.5:  # More than 2.5 standard deviations
                    anomalies.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'cost': metric.cost,
                        'z_score': z_score,
                        'department': metric.department,
                        'project': metric.project
                    })
                    
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)
        
    def _store_roi_analysis(self, roi_analysis: ROIAnalysis):
        """Store ROI analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO roi_analysis
            (timestamp, investment, return_value, roi_percentage, 
             payback_period_days, net_present_value, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            roi_analysis.investment,
            roi_analysis.return_value,
            roi_analysis.roi_percentage,
            roi_analysis.payback_period_days,
            roi_analysis.net_present_value,
            roi_analysis.confidence_score
        ))
        
        conn.commit()
        conn.close()
        
    def _store_utilization_report(self, report: UtilizationReport):
        """Store utilization report in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO utilization_reports
            (timestamp, resource_type, current_utilization, peak_utilization,
             avg_utilization, efficiency_score, optimization_potential, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            report.resource_type,
            report.current_utilization,
            report.peak_utilization,
            report.avg_utilization,
            report.efficiency_score,
            report.optimization_potential,
            json.dumps(report.recommendations)
        ))
        
        conn.commit()
        conn.close()

# Example usage and testing
if __name__ == "__main__":
    # Initialize BI engine
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate sample business metrics
    sample_metrics = []
    base_time = datetime.now() - timedelta(days=30)
    
    departments = ['engineering', 'marketing', 'sales', 'operations']
    projects = ['project_a', 'project_b', 'project_c']
    
    for i in range(1000):
        timestamp = base_time + timedelta(hours=i)
        
        metric = BusinessMetric(
            timestamp=timestamp,
            metric_name=f"cpu_utilization_{i%4}",
            value=np.random.normal(0.65, 0.15),  # Target around 65%
            cost=np.random.normal(50, 15),       # Average $50/hour
            department=np.random.choice(departments),
            project=np.random.choice(projects),
            tags={'environment': 'production', 'region': 'us-west'}
        )
        
        sample_metrics.append(metric)
        bi_engine.add_business_metric(metric)
        
    # Generate comprehensive business intelligence reports
    print("🔍 Generating Business Intelligence Reports...")
    
    # ROI Analysis
    roi_analysis = bi_engine.calculate_roi_analysis(
        investment=100000,
        revenue_stream=[15000] * 12  # $15k per month for 12 months
    )
    print(f"📊 ROI Analysis: {roi_analysis.roi_percentage:.1f}% ROI")
    
    # Cost Breakdown
    cost_breakdown = bi_engine.generate_cost_breakdown(sample_metrics)
    print(f"💰 Total Cost: ${cost_breakdown.total_cost:,.2f}")
    print(f"📈 Cost Trend: {cost_breakdown.trends.get('total', 0):.1f}%")
    
    # Resource Utilization
    cpu_utilization = bi_engine.analyze_system_utilization(sample_metrics, 'cpu')
    print(f"🔧 CPU Efficiency: {cpu_utilization.efficiency_score:.1%}")
    
    # Executive Dashboard
    dashboard = bi_engine.generate_executive_dashboard(sample_metrics)
    print(f"🎯 Performance Score: {dashboard['kpis']['performance_score']:.1f}/100")
    print(f"💡 Business Insights: {len(dashboard['business_insights'])} insights generated")
    print(f"📋 Recommendations: {len(dashboard['recommendations'])} executive recommendations")
    
    print("\n✅ Business Intelligence Engine - Production Ready!")