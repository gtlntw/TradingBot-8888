"""
Feature selection module for reducing feature dimensionality.
Implements correlation-based and importance-based feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import save_json


class FeatureSelector(LoggerMixin):
    """Feature selection utility for ML models."""

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.001,
        max_features: int = 30
    ):
        """
        Initialize feature selector.

        Args:
            correlation_threshold: Remove features with correlation above this
            importance_threshold: Minimum importance score to keep feature
            max_features: Maximum number of features to select
        """
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.max_features = max_features
        self.selected_features = None
        self.feature_importance = None
        self.correlation_matrix = None
        self.removed_features = {}

    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Tuple of (filtered X, remaining feature names)
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        self.logger.info(f"Removing features with correlation > {self.correlation_threshold}")

        # Calculate correlation matrix
        self.correlation_matrix = X.corr().abs()

        # Find pairs of highly correlated features
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix), k=1).astype(bool)
        to_remove = set()

        for i in range(len(self.correlation_matrix)):
            for j in range(i+1, len(self.correlation_matrix)):
                if self.correlation_matrix.iloc[i, j] > self.correlation_threshold:
                    # Keep the feature that appears first (arbitrary choice)
                    feature_to_remove = self.correlation_matrix.columns[j]
                    feature_to_keep = self.correlation_matrix.columns[i]
                    to_remove.add(feature_to_remove)

                    if feature_to_remove not in self.removed_features:
                        self.removed_features[feature_to_remove] = []
                    self.removed_features[feature_to_remove].append({
                        'reason': 'high_correlation',
                        'correlated_with': feature_to_keep,
                        'correlation': float(self.correlation_matrix.iloc[i, j])
                    })

        remaining_features = [f for f in feature_names if f not in to_remove]

        self.logger.info(f"Removed {len(to_remove)} correlated features: {sorted(to_remove)}")
        self.logger.info(f"Remaining features: {len(remaining_features)}")

        return X[remaining_features], remaining_features

    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'classification'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features based on Random Forest feature importance.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            model_type: 'classification' or 'regression'

        Returns:
            Tuple of (filtered X, selected feature names)
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        self.logger.info(f"Calculating feature importance using Random Forest")

        # Train Random Forest to get feature importance
        if model_type == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        rf.fit(X, y)

        # Get feature importance
        importance = rf.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance))

        # Sort features by importance
        sorted_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top features
        selected_features = []
        for feat_name, feat_importance in sorted_features:
            if feat_importance >= self.importance_threshold and len(selected_features) < self.max_features:
                selected_features.append(feat_name)
            else:
                if feat_name not in self.removed_features:
                    self.removed_features[feat_name] = []
                self.removed_features[feat_name].append({
                    'reason': 'low_importance',
                    'importance': float(feat_importance)
                })

        self.logger.info(f"Selected {len(selected_features)} features based on importance")
        self.logger.info(f"Top 10 features by importance:")
        for feat_name, feat_importance in sorted_features[:10]:
            self.logger.info(f"  {feat_name}: {feat_importance:.4f}")

        return X[selected_features], selected_features

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'classification'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete feature selection pipeline.

        1. Remove highly correlated features
        2. Select by feature importance

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            model_type: 'classification' or 'regression'

        Returns:
            Tuple of (filtered X, selected feature names)
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        self.logger.info(f"Starting feature selection with {len(feature_names)} features")

        # Step 1: Remove correlated features
        X_uncorr, uncorr_features = self.remove_correlated_features(X, feature_names)

        # Step 2: Select by importance
        X_selected, selected_features = self.select_by_importance(
            X_uncorr, y, uncorr_features, model_type
        )

        self.selected_features = selected_features

        self.logger.info(f"Feature selection complete: {len(feature_names)} → {len(selected_features)} features")

        return X_selected, selected_features

    def plot_correlation_heatmap(self, output_path: Path, top_n: int = 50):
        """Plot correlation heatmap of top features."""
        if self.correlation_matrix is None:
            raise ValueError("Must run remove_correlated_features first")

        plt.figure(figsize=(16, 14))

        # Select top N features for readability
        if len(self.correlation_matrix) > top_n:
            # Get features with highest average correlation
            avg_corr = self.correlation_matrix.abs().mean().sort_values(ascending=False)
            top_features = avg_corr.head(top_n).index
            corr_subset = self.correlation_matrix.loc[top_features, top_features]
        else:
            corr_subset = self.correlation_matrix

        sns.heatmap(
            corr_subset,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title(f'Feature Correlation Heatmap (Top {len(corr_subset)} Features)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved correlation heatmap to {output_path}")

    def plot_feature_importance(self, output_path: Path, top_n: int = 30):
        """Plot feature importance scores."""
        if self.feature_importance is None:
            raise ValueError("Must run select_by_importance first")

        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features, importance = zip(*sorted_features)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features by Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved feature importance plot to {output_path}")

    def save_results(self, output_dir: Path):
        """Save feature selection results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save selected features
        save_json({
            'selected_features': self.selected_features,
            'num_selected': len(self.selected_features) if self.selected_features else 0,
            'removed_features': self.removed_features,
            'feature_importance': self.feature_importance,
            'parameters': {
                'correlation_threshold': self.correlation_threshold,
                'importance_threshold': self.importance_threshold,
                'max_features': self.max_features
            }
        }, output_dir / 'feature_selection_results.json')

        # Save plots if data available
        if self.correlation_matrix is not None:
            self.plot_correlation_heatmap(output_dir / 'correlation_heatmap.png')

        if self.feature_importance is not None:
            self.plot_feature_importance(output_dir / 'feature_importance.png')

        self.logger.info(f"Saved feature selection results to {output_dir}")
