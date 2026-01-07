"""
Sequence generation utilities for time series models.
"""
import numpy as np
from typing import Tuple, Optional


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 20,
    return_indices: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from 2D feature array for LSTM/Transformer models.

    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of targets (n_samples,)
        sequence_length: Number of timesteps in each sequence
        return_indices: If True, also return the indices used

    Returns:
        X_seq: 3D array of shape (n_sequences, sequence_length, n_features)
        y_seq: 1D array of targets (n_sequences,)
        indices (optional): Array of sequence end indices

    Example:
        >>> X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])  # 5 samples, 2 features
        >>> y = np.array([0, 1, 2, 3, 4])
        >>> X_seq, y_seq = create_sequences(X, y, sequence_length=3)
        >>> X_seq.shape  # (3, 3, 2) - 3 sequences of 3 timesteps with 2 features
        >>> y_seq  # [2, 3, 4] - targets for each sequence
    """
    if len(X.shape) != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")

    if len(y.shape) != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")

    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    if sequence_length > X.shape[0]:
        raise ValueError(
            f"sequence_length ({sequence_length}) cannot be larger than "
            f"number of samples ({X.shape[0]})"
        )

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_sequences = n_samples - sequence_length + 1

    # Pre-allocate arrays for efficiency
    X_seq = np.zeros((n_sequences, sequence_length, n_features), dtype=X.dtype)
    y_seq = np.zeros(n_sequences, dtype=y.dtype)
    indices = np.zeros(n_sequences, dtype=np.int32)

    # Create sliding window sequences
    for i in range(n_sequences):
        X_seq[i] = X[i:i + sequence_length]
        y_seq[i] = y[i + sequence_length - 1]  # Target is last timestep's label
        indices[i] = i + sequence_length - 1

    if return_indices:
        return X_seq, y_seq, indices
    else:
        return X_seq, y_seq


def create_sequences_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sequence_length: int = 20
) -> Tuple:
    """
    Create sequences for both training and validation sets.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        sequence_length: Number of timesteps

    Returns:
        If validation data provided:
            X_train_seq, y_train_seq, X_val_seq, y_val_seq
        Otherwise:
            X_train_seq, y_train_seq, None, None
    """
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)

    if X_val is not None and y_val is not None:
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        return X_train_seq, y_train_seq, X_val_seq, y_val_seq
    else:
        return X_train_seq, y_train_seq, None, None
