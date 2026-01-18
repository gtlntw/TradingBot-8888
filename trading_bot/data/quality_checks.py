"""
Data quality validation checks for market data.
Detects flash crashes, gaps, anomalies, and source divergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import save_json


class DataQualityChecker(LoggerMixin):
    """Comprehensive data quality validation."""

    def __init__(
        self,
        flash_crash_threshold: float = 0.10,  # 10% move in 1 period
        gap_threshold_hours: int = 48,  # Max acceptable gap
        outlier_std_threshold: float = 5.0,  # Standard deviations
        volume_spike_threshold: float = 10.0  # 10x normal volume
    ):
        """
        Initialize data quality checker.

        Args:
            flash_crash_threshold: Price move % to flag as flash crash
            gap_threshold_hours: Maximum acceptable data gap in hours
            outlier_std_threshold: Standard deviations for outlier detection
            volume_spike_threshold: Volume multiplier for spike detection
        """
        self.flash_crash_threshold = flash_crash_threshold
        self.gap_threshold_hours = gap_threshold_hours
        self.outlier_std_threshold = outlier_std_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.issues = {
            'flash_crashes': [],
            'data_gaps': [],
            'extreme_outliers': [],
            'volume_anomalies': [],
            'source_divergence': [],
            'data_freshness': []
        }

    def detect_flash_crashes(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect flash crashes (sudden extreme price moves).

        Args:
            data: DataFrame with OHLC data

        Returns:
            List of detected flash crashes
        """
        self.logger.info("Checking for flash crashes...")

        crashes = []

        # Calculate returns
        returns = data['close'].pct_change()

        # Find extreme moves
        crash_mask = np.abs(returns) > self.flash_crash_threshold

        for idx in data[crash_mask].index:
            crash = {
                'date': str(idx),
                'price_before': float(data.loc[idx, 'close'] / (1 + returns.loc[idx])),
                'price_after': float(data.loc[idx, 'close']),
                'return': float(returns.loc[idx]),
                'magnitude': float(abs(returns.loc[idx])),
                'direction': 'crash' if returns.loc[idx] < 0 else 'spike'
            }
            crashes.append(crash)

        self.issues['flash_crashes'] = crashes

        if crashes:
            self.logger.warning(f"Found {len(crashes)} flash crashes/spikes")
            for crash in crashes[:3]:  # Show first 3
                self.logger.warning(
                    f"  {crash['date']}: {crash['magnitude']*100:.1f}% {crash['direction']}"
                )
        else:
            self.logger.info("No flash crashes detected")

        return crashes

    def detect_data_gaps(self, data: pd.DataFrame, expected_interval: str = '1D') -> List[Dict]:
        """
        Detect gaps in time series data.

        Args:
            data: DataFrame with DatetimeIndex
            expected_interval: Expected data interval ('1D', '1H', etc.)

        Returns:
            List of detected gaps
        """
        self.logger.info("Checking for data gaps...")

        gaps = []

        # Calculate time differences
        time_diffs = data.index.to_series().diff()

        # Convert expected interval to timedelta
        if expected_interval == '1D':
            expected_delta = timedelta(days=1)
            max_gap = timedelta(hours=self.gap_threshold_hours)
        elif expected_interval == '1H':
            expected_delta = timedelta(hours=1)
            max_gap = timedelta(hours=self.gap_threshold_hours)
        else:
            expected_delta = pd.Timedelta(expected_interval)
            max_gap = expected_delta * 48

        # Find gaps larger than threshold
        gap_mask = time_diffs > max_gap

        for idx in data[gap_mask].index:
            gap_size = time_diffs.loc[idx]
            prev_idx = data.index[data.index < idx][-1]

            gap = {
                'start_date': str(prev_idx),
                'end_date': str(idx),
                'gap_duration': str(gap_size),
                'gap_hours': gap_size.total_seconds() / 3600,
                'missing_periods': int(gap_size / expected_delta) - 1
            }
            gaps.append(gap)

        self.issues['data_gaps'] = gaps

        if gaps:
            self.logger.warning(f"Found {len(gaps)} data gaps")
            for gap in gaps[:3]:  # Show first 3
                self.logger.warning(
                    f"  {gap['start_date']} → {gap['end_date']}: "
                    f"{gap['gap_hours']:.1f} hours ({gap['missing_periods']} missing periods)"
                )
        else:
            self.logger.info("No significant data gaps detected")

        return gaps

    def detect_extreme_outliers(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect extreme outliers using statistical methods.

        Args:
            data: DataFrame with price data

        Returns:
            List of detected outliers
        """
        self.logger.info("Checking for extreme outliers...")

        outliers = []

        for col in ['high', 'low', 'close']:
            if col not in data.columns:
                continue

            # Calculate z-scores
            mean = data[col].mean()
            std = data[col].std()
            z_scores = np.abs((data[col] - mean) / std)

            # Find outliers
            outlier_mask = z_scores > self.outlier_std_threshold

            for idx in data[outlier_mask].index:
                outlier = {
                    'date': str(idx),
                    'column': col,
                    'value': float(data.loc[idx, col]),
                    'z_score': float(z_scores.loc[idx]),
                    'std_deviation': float((data.loc[idx, col] - mean) / std)
                }
                outliers.append(outlier)

        self.issues['extreme_outliers'] = outliers

        if outliers:
            self.logger.warning(f"Found {len(outliers)} extreme outliers")
            for outlier in outliers[:3]:
                self.logger.warning(
                    f"  {outlier['date']} {outlier['column']}: "
                    f"{outlier['value']:.2f} (z={outlier['z_score']:.1f})"
                )
        else:
            self.logger.info("No extreme outliers detected")

        return outliers

    def detect_volume_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect volume spikes and anomalies.

        Args:
            data: DataFrame with volume data

        Returns:
            List of detected volume anomalies
        """
        self.logger.info("Checking for volume anomalies...")

        anomalies = []

        if 'volume' not in data.columns:
            self.logger.warning("No volume data available")
            return anomalies

        # Calculate rolling average volume (30-day window)
        avg_volume = data['volume'].rolling(window=30, min_periods=1).mean()

        # Find volume spikes
        volume_ratio = data['volume'] / avg_volume
        spike_mask = volume_ratio > self.volume_spike_threshold

        for idx in data[spike_mask].index:
            anomaly = {
                'date': str(idx),
                'volume': float(data.loc[idx, 'volume']),
                'avg_volume': float(avg_volume.loc[idx]),
                'spike_ratio': float(volume_ratio.loc[idx]),
                'magnitude': f"{volume_ratio.loc[idx]:.1f}x normal"
            }
            anomalies.append(anomaly)

        # Also check for suspiciously low volume (near zero)
        low_volume_mask = data['volume'] < (avg_volume * 0.01)  # < 1% of average

        for idx in data[low_volume_mask].index:
            anomaly = {
                'date': str(idx),
                'volume': float(data.loc[idx, 'volume']),
                'avg_volume': float(avg_volume.loc[idx]),
                'spike_ratio': float(volume_ratio.loc[idx]),
                'magnitude': 'suspiciously low'
            }
            anomalies.append(anomaly)

        self.issues['volume_anomalies'] = anomalies

        if anomalies:
            self.logger.warning(f"Found {len(anomalies)} volume anomalies")
            for anomaly in anomalies[:3]:
                self.logger.warning(
                    f"  {anomaly['date']}: {anomaly['volume']:.0f} ({anomaly['magnitude']})"
                )
        else:
            self.logger.info("No volume anomalies detected")

        return anomalies

    def check_source_divergence(
        self,
        source_data: Dict[str, pd.DataFrame],
        divergence_threshold: float = 0.05  # 5% price difference
    ) -> List[Dict]:
        """
        Check for divergence between multiple data sources.

        Args:
            source_data: Dictionary of {source_name: DataFrame}
            divergence_threshold: Price difference threshold to flag

        Returns:
            List of detected divergences
        """
        self.logger.info("Checking for source divergence...")

        divergences = []

        if len(source_data) < 2:
            self.logger.info("Need at least 2 sources to check divergence")
            return divergences

        # Get common dates
        source_names = list(source_data.keys())
        common_dates = source_data[source_names[0]].index

        for source_name in source_names[1:]:
            common_dates = common_dates.intersection(source_data[source_name].index)

        # Compare prices at each common date
        for date in common_dates:
            prices = {}
            for source_name, df in source_data.items():
                if date in df.index:
                    prices[source_name] = df.loc[date, 'close']

            if len(prices) < 2:
                continue

            # Calculate price differences
            price_values = list(prices.values())
            max_price = max(price_values)
            min_price = min(price_values)

            if max_price > 0:
                divergence_pct = (max_price - min_price) / max_price

                if divergence_pct > divergence_threshold:
                    divergence = {
                        'date': str(date),
                        'sources': prices,
                        'max_price': float(max_price),
                        'min_price': float(min_price),
                        'divergence_pct': float(divergence_pct),
                        'divergence_amount': float(max_price - min_price)
                    }
                    divergences.append(divergence)

        self.issues['source_divergence'] = divergences

        if divergences:
            self.logger.warning(f"Found {len(divergences)} source divergence issues")
            for div in divergences[:3]:
                self.logger.warning(
                    f"  {div['date']}: {div['divergence_pct']*100:.1f}% difference "
                    f"(${div['divergence_amount']:.2f})"
                )
        else:
            self.logger.info("No significant source divergence detected")

        return divergences

    def check_data_freshness(
        self,
        data: pd.DataFrame,
        max_age_hours: int = 48
    ) -> Dict:
        """
        Check if data is recent enough.

        Args:
            data: DataFrame with DatetimeIndex
            max_age_hours: Maximum acceptable age in hours

        Returns:
            Freshness check result
        """
        self.logger.info("Checking data freshness...")

        latest_date = data.index[-1]
        # Handle timezone-aware timestamps
        if latest_date.tzinfo is not None:
            now = pd.Timestamp.now(tz=latest_date.tzinfo)
        else:
            now = pd.Timestamp.now()
        age = now - latest_date
        age_hours = age.total_seconds() / 3600

        freshness = {
            'latest_date': str(latest_date),
            'current_time': str(now),
            'age_hours': float(age_hours),
            'is_fresh': age_hours <= max_age_hours,
            'max_age_hours': max_age_hours
        }

        self.issues['data_freshness'] = [freshness]

        if not freshness['is_fresh']:
            self.logger.warning(
                f"Data is stale: {age_hours:.1f} hours old "
                f"(max: {max_age_hours} hours)"
            )
        else:
            self.logger.info(f"Data is fresh: {age_hours:.1f} hours old")

        return freshness

    def run_all_checks(
        self,
        data: pd.DataFrame,
        source_data: Optional[Dict[str, pd.DataFrame]] = None,
        expected_interval: str = '1D'
    ) -> Dict:
        """
        Run all quality checks.

        Args:
            data: Main DataFrame to check
            source_data: Optional dict of source DataFrames for divergence check
            expected_interval: Expected data interval

        Returns:
            Dictionary of all issues found
        """
        self.logger.info("Running comprehensive data quality checks...")

        # Run all checks
        self.detect_flash_crashes(data)
        self.detect_data_gaps(data, expected_interval)
        self.detect_extreme_outliers(data)
        self.detect_volume_anomalies(data)
        self.check_data_freshness(data)

        if source_data and len(source_data) > 1:
            self.check_source_divergence(source_data)

        # Summary
        total_issues = sum(
            len(issues) if isinstance(issues, list) else 1
            for issues in self.issues.values()
            if issues
        )

        self.logger.info(f"Quality check complete: {total_issues} total issues found")

        return self.issues

    def get_summary(self) -> Dict:
        """Get summary of issues."""
        summary = {}

        for check_name, issues in self.issues.items():
            if isinstance(issues, list):
                summary[check_name] = len(issues)
            else:
                summary[check_name] = 1 if issues else 0

        summary['total_issues'] = sum(summary.values())

        return summary

    def save_report(self, output_path: Path):
        """Save quality check report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'summary': self.get_summary(),
            'issues': self.issues,
            'thresholds': {
                'flash_crash_threshold': self.flash_crash_threshold,
                'gap_threshold_hours': self.gap_threshold_hours,
                'outlier_std_threshold': self.outlier_std_threshold,
                'volume_spike_threshold': self.volume_spike_threshold
            }
        }

        save_json(report, output_path)
        self.logger.info(f"Saved quality report to {output_path}")

        return report
