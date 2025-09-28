"""
Helper utilities for the Bitcoin trading bot.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


def setup_directories(directories: List[str]) -> None:
    """
    Create directories if they don't exist.

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["data", "models", "trading"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False

    # Validate data section
    data_config = config.get("data", {})
    if not data_config.get("sources"):
        logger.error("No data sources specified")
        return False

    if not data_config.get("symbols"):
        logger.error("No trading symbols specified")
        return False

    # Validate models section
    models_config = config.get("models", {})
    if not models_config.get("algorithms"):
        logger.error("No ML algorithms specified")
        return False

    logger.info("Configuration validation passed")
    return True


def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to file using pickle.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    logger.debug(f"Saved object to {filepath}")


def load_object(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    logger.debug(f"Loaded object from {filepath}")
    return obj


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {filepath}")
    return data


def generate_timestamp() -> str:
    """
    Generate timestamp string for file naming.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns for a price series.

    Args:
        prices: Price series
        periods: Number of periods for return calculation

    Returns:
        Returns series
    """
    return prices.pct_change(periods=periods)


def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate rolling volatility of returns.

    Args:
        returns: Returns series
        window: Rolling window size

    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annual)

    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Price series

    Returns:
        Maximum drawdown as percentage
    """
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """
    Generate list of dates between start and end date.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of date strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def ensure_dataframe_index(df: pd.DataFrame, date_column: str = 'timestamp') -> pd.DataFrame:
    """
    Ensure DataFrame has datetime index.

    Args:
        df: Input DataFrame
        date_column: Name of date/time column

    Returns:
        DataFrame with datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_column in df.columns:
            df = df.set_index(pd.to_datetime(df[date_column]))
            df = df.drop(columns=[date_column])
        else:
            logger.warning(f"Date column '{date_column}' not found, assuming index is datetime")
            df.index = pd.to_datetime(df.index)

    return df.sort_index()


def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Split DataFrame into train and test sets for time series.

    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing

    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
    return train_df, test_df