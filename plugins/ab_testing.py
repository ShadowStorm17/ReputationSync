"""
A/B testing plugin.
Enables testing of reputation strategies and measuring their impact.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
import random
import numpy as np
from datetime import datetime, timedelta
from app.core.plugins.base import ABTestingPlugin, PluginType, PluginMetadata
from app.core.error_handling import ReputationError, ErrorSeverity, ErrorCategory

class VariantData(TypedDict):
    """Type definition for variant data."""
    name: str
    description: str
    config: Dict[str, Any]

class ExperimentData(TypedDict):
    """Type definition for experiment data."""
    name: str
    description: str
    variants: List[VariantData]
    metrics: List[str]
    start_date: datetime
    end_date: datetime
    status: str
    assignments: Dict[str, VariantData]
    results: Dict[str, Dict[str, List[float]]]

class ReputationABTesting(ABTestingPlugin):
    """Plugin for A/B testing reputation strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize A/B testing plugin."""
        super().__init__(config)
        self.experiments: Dict[str, ExperimentData] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="reputation_ab_testing",
            version="1.0.0",
            description="A/B testing plugin for reputation strategies",
            author="Reputation Sync Team",
            type=PluginType.A_B_TESTING,
            config_schema={
                "type": "object",
                "properties": {
                    "min_sample_size": {
                        "type": "integer",
                        "description": "Minimum sample size for statistical significance",
                        "minimum": 30
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level for statistical tests",
                        "minimum": 0.5,
                        "maximum": 0.99,
                        "default": 0.95
                    },
                    "max_experiment_duration": {
                        "type": "integer",
                        "description": "Maximum experiment duration in days",
                        "minimum": 1
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to track during experiments"
                    },
                    "random_seed": {
                        "type": "integer",
                        "description": "Random seed for variant assignment",
                        "default": 42
                    }
                }
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize plugin."""
        try:
            # Set random seed
            random.seed(self.config.get("random_seed", 42))
            np.random.seed(self.config.get("random_seed", 42))
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error initializing A/B testing plugin: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        return True
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        metrics: List[str],
        duration_days: int
    ) -> str:
        """Create a new A/B test experiment."""
        try:
            # Validate duration
            if duration_days > self.config.get("max_experiment_duration", 30):
                raise ReputationError(
                    message="Experiment duration exceeds maximum allowed",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Generate experiment ID
            experiment_id = f"exp_{len(self.experiments)}"
            
            # Validate variants
            if len(variants) < 2:
                raise ReputationError(
                    message="At least two variants required for A/B testing",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Create experiment
            self.experiments[experiment_id] = {
                "name": name,
                "description": description,
                "variants": variants,
                "metrics": metrics,
                "start_date": datetime.now(),
                "end_date": datetime.now() + timedelta(days=duration_days),
                "status": "active",
                "assignments": {},
                "results": {metric: {variant["name"]: [] for variant in variants}
                          for metric in metrics}
            }
            
            return experiment_id
            
        except Exception as e:
            raise ReputationError(
                message=f"Error creating experiment: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Assign a variant to a user."""
        try:
            if experiment_id not in self.experiments:
                raise ReputationError(
                    message="Experiment not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            experiment = self.experiments[experiment_id]
            
            # Check if experiment is active
            if experiment["status"] != "active":
                raise ReputationError(
                    message="Experiment is not active",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check if user already assigned
            if user_id in experiment["assignments"]:
                return experiment["assignments"][user_id]
            
            # Assign variant randomly
            variant = random.choice(experiment["variants"])
            experiment["assignments"][user_id] = variant
            
            return variant
            
        except Exception as e:
            raise ReputationError(
                message=f"Error assigning variant: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric: str,
        value: float
    ) -> bool:
        """Record a metric value for a user in an experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ReputationError(
                    message="Experiment not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            experiment = self.experiments[experiment_id]
            
            # Check if experiment is active
            if experiment["status"] != "active":
                raise ReputationError(
                    message="Experiment is not active",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check if metric is being tracked
            if metric not in experiment["metrics"]:
                raise ReputationError(
                    message="Metric not being tracked in experiment",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check if user is assigned
            if user_id not in experiment["assignments"]:
                raise ReputationError(
                    message="User not assigned to experiment",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Record metric
            variant = experiment["assignments"][user_id]
            experiment["results"][metric][variant["name"]].append(value)
            
            return True
            
        except Exception as e:
            raise ReputationError(
                message=f"Error recording metric: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_experiment_results(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Get results for an experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ReputationError(
                    message="Experiment not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            experiment = self.experiments[experiment_id]
            results = {
                "experiment_id": experiment_id,
                "name": experiment["name"],
                "description": experiment["description"],
                "status": experiment["status"],
                "start_date": experiment["start_date"],
                "end_date": experiment["end_date"],
                "metrics": {}
            }
            
            # Calculate statistics for each metric
            for metric in experiment["metrics"]:
                metric_results = {}
                
                for variant in experiment["variants"]:
                    values = experiment["results"][metric][variant["name"]]
                    
                    if values:
                        metric_results[variant["name"]] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "count": len(values),
                            "min": np.min(values),
                            "max": np.max(values)
                        }
                    
                    # Calculate statistical significance
                    if len(experiment["variants"]) == 2:
                        control_values = experiment["results"][metric][experiment["variants"][0]["name"]]
                        treatment_values = experiment["results"][metric][experiment["variants"][1]["name"]]
                        
                        if control_values and treatment_values:
                            t_stat, p_value = self._calculate_t_test(control_values, treatment_values)
                            metric_results["statistical_significance"] = {
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "significant": p_value < (1 - self.config.get("confidence_level", 0.95))
                            }
                
                results["metrics"][metric] = metric_results
            
            return results
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting experiment results: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def end_experiment(
        self,
        experiment_id: str
    ) -> bool:
        """End an experiment and calculate final results."""
        try:
            if experiment_id not in self.experiments:
                raise ReputationError(
                    message="Experiment not found",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            experiment = self.experiments[experiment_id]
            
            # Check if experiment is already ended
            if experiment["status"] != "active":
                raise ReputationError(
                    message="Experiment is already ended",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # End experiment
            experiment["status"] = "ended"
            experiment["end_date"] = datetime.now()
            
            # Calculate final results
            results = await self.get_experiment_results(experiment_id)
            self.results[experiment_id] = results
            
            return True
            
        except Exception as e:
            raise ReputationError(
                message=f"Error ending experiment: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    def _calculate_t_test(
        self,
        control_values: List[float],
        treatment_values: List[float]
    ) -> tuple:
        """Calculate t-test between control and treatment groups."""
        try:
            # Perform t-test
            t_stat, p_value = np.stats.ttest_ind(control_values, treatment_values)
            return t_stat, p_value
            
        except Exception as e:
            raise ReputationError(
                message=f"Error calculating t-test: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            ) 