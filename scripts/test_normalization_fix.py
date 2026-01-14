#!/usr/bin/env python3
"""
Test that sequence normalization fixes the degenerate model issue.
"""

import sys
sys.path.insert(0, '/home/user/TradingBot-8888')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from trading_bot.data.sequences import SequenceGenerator
from trading_bot.models.sequence_models import SequenceLSTMModel

def main():
    print("=" * 80)
    print("TESTING NORMALIZATION FIX")
    print("=" * 80)

    # Collect data
    print("\n1. Collecting data...")
    ticker = yf.Ticker('BTC-USD')
    data = ticker.history(period='400d', interval='1d')
    data = data.reset_index()
    data.columns = [c.lower() for c in data.columns]
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    data = data.set_index('date')

    # Create target
    future_return = data['close'].shift(-1) / data['close'] - 1
    data['profitable_trade'] = (future_return > 0).astype(int)
    data = data.dropna()

    # Split data
    train_size = 365
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:train_size+60].copy()
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")

    # Create sequences
    print("\n2. Creating sequences...")
    seq_gen = SequenceGenerator(sequence_length=30, target_horizon=1)

    X_train_seq, y_train_seq, _ = seq_gen.create_sequences(
        train_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )

    X_test_seq, y_test_seq, _ = seq_gen.create_sequences(
        test_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )

    print(f"   Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    print(f"   Train range: min={X_train_seq.min():.2f}, max={X_train_seq.max():.2f}")
    print(f"   Test range: min={X_test_seq.min():.2f}, max={X_test_seq.max():.2f}")

    # Normalize sequences
    print("\n3. Normalizing sequences...")
    original_train_shape = X_train_seq.shape
    original_test_shape = X_test_seq.shape

    X_train_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[2])
    X_test_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[2])

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_test_normalized = scaler.transform(X_test_reshaped)

    X_train_normalized = X_train_normalized.reshape(original_train_shape)
    X_test_normalized = X_test_normalized.reshape(original_test_shape)

    print(f"   Normalized train range: min={X_train_normalized.min():.4f}, max={X_train_normalized.max():.4f}")
    print(f"   Normalized test range: min={X_test_normalized.min():.4f}, max={X_test_normalized.max():.4f}")

    # Train LSTM with normalized data
    print("\n4. Training LSTM with NORMALIZED data...")
    lstm = SequenceLSTMModel(
        model_type='classification',
        params={
            'units': 64,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 5,
            'batch_size': 32,
            'patience': 3
        }
    )
    lstm.fit(X_train_normalized, y_train_seq)

    # Predict
    print("\n5. Making predictions...")
    predictions = lstm.predict(X_test_normalized)
    probabilities = lstm.predict_proba(X_test_normalized)

    print(f"   Prediction distribution: {np.bincount(predictions.astype(int))}")
    print(f"   Unique predictions: {np.unique(predictions)}")
    print(f"\n   First 10 predictions:")
    for i in range(min(10, len(predictions))):
        print(f"      Sample {i}: [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}] → pred={predictions[i]} (actual={int(y_test_seq[i])})")

    # Check if all probabilities are identical
    prob_std = probabilities.std(axis=0)
    print(f"\n   Probability std dev: [class 0: {prob_std[0]:.6f}, class 1: {prob_std[1]:.6f}]")

    if prob_std[0] < 0.001 and prob_std[1] < 0.001:
        print(f"   ❌ FAILED: Probabilities still constant!")
        return False
    else:
        print(f"   ✅ SUCCESS: Probabilities are diverse!")

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test_seq)
    print(f"\n   Accuracy: {accuracy:.2%}")

    # Check if predictions are diverse
    num_unique = len(np.unique(predictions))
    if num_unique == 1:
        print(f"   ❌ FAILED: All predictions are {predictions[0]}")
        return False
    else:
        print(f"   ✅ SUCCESS: {num_unique} unique predictions")

    print("\n" + "=" * 80)
    print("NORMALIZATION FIX VERIFIED!")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
