#!/usr/bin/env python3
"""
Test different normalization methods for sequences.
"""

import sys
sys.path.insert(0, '/home/user/TradingBot-8888')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from trading_bot.data.sequences import SequenceGenerator
from trading_bot.models.sequence_models import SequenceLSTMModel

def test_method(X_train, X_test, y_train, y_test, method_name, normalize_func):
    """Test a normalization method."""
    print(f"\n{'='*80}")
    print(f"Testing: {method_name}")
    print(f"{'='*80}")

    X_train_norm, X_test_norm = normalize_func(X_train, X_test)

    print(f"Train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(f"Test range: [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")

    # Train LSTM
    lstm = SequenceLSTMModel(
        model_type='classification',
        params={
            'units': 32,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'patience': 5
        }
    )
    lstm.fit(X_train_norm, y_train)

    # Predict
    predictions = lstm.predict(X_test_norm)
    probabilities = lstm.predict_proba(X_test_norm)

    # Calculate diversity metrics
    prob_std = probabilities.std(axis=0)
    num_unique = len(np.unique(predictions))
    accuracy = np.mean(predictions == y_test)

    print(f"\nResults:")
    print(f"  Predictions: {np.bincount(predictions.astype(int))}")
    print(f"  Unique: {num_unique}")
    print(f"  Prob std: [class 0: {prob_std[0]:.6f}, class 1: {prob_std[1]:.6f}]")
    print(f"  Accuracy: {accuracy:.2%}")

    # Show some predictions
    print(f"\nFirst 10 predictions:")
    for i in range(min(10, len(predictions))):
        print(f"  Sample {i}: [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}] → {predictions[i]} (actual={int(y_test[i])})")

    return prob_std[0] > 0.01, num_unique > 1

def main():
    # Collect data
    print("Collecting data...")
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
    train_size = 300
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:train_size+30].copy()

    # Create sequences
    seq_gen = SequenceGenerator(sequence_length=30, target_horizon=1)
    X_train, y_train, _ = seq_gen.create_sequences(
        train_data, ['open', 'high', 'low', 'close', 'volume'], 'profitable_trade'
    )
    X_test, y_test, _ = seq_gen.create_sequences(
        test_data, ['open', 'high', 'low', 'close', 'volume'], 'profitable_trade'
    )

    print(f"Shapes: train={X_train.shape}, test={X_test.shape}")
    print(f"Original range: train=[{X_train.min():.0f}, {X_train.max():.0f}]")

    # Method 1: Global MinMax (current approach)
    def method1_global_minmax(X_tr, X_te):
        scaler = MinMaxScaler()
        X_tr_flat = X_tr.reshape(-1, X_tr.shape[2])
        X_te_flat = X_te.reshape(-1, X_te.shape[2])
        X_tr_norm = scaler.fit_transform(X_tr_flat).reshape(X_tr.shape)
        X_te_norm = scaler.transform(X_te_flat).reshape(X_te.shape)
        return X_tr_norm, X_te_norm

    # Method 2: Per-sequence MinMax
    def method2_per_sequence_minmax(X_tr, X_te):
        X_tr_norm = np.zeros_like(X_tr)
        X_te_norm = np.zeros_like(X_te)

        for i in range(X_tr.shape[0]):
            scaler = MinMaxScaler()
            X_tr_norm[i] = scaler.fit_transform(X_tr[i])

        for i in range(X_te.shape[0]):
            scaler = MinMaxScaler()
            X_te_norm[i] = scaler.fit_transform(X_te[i])

        return X_tr_norm, X_te_norm

    # Method 3: StandardScaler (Z-score)
    def method3_standardscaler(X_tr, X_te):
        scaler = StandardScaler()
        X_tr_flat = X_tr.reshape(-1, X_tr.shape[2])
        X_te_flat = X_te.reshape(-1, X_te.shape[2])
        X_tr_norm = scaler.fit_transform(X_tr_flat).reshape(X_tr.shape)
        X_te_norm = scaler.transform(X_te_flat).reshape(X_te.shape)
        return X_tr_norm, X_te_norm

    # Method 4: Returns-based (percentage changes)
    def method4_returns_based(X_tr, X_te):
        X_tr_norm = np.zeros_like(X_tr)
        X_te_norm = np.zeros_like(X_te)

        # Calculate percentage changes within each sequence
        for i in range(X_tr.shape[0]):
            seq = X_tr[i]
            # For each feature, calculate pct change from first value
            X_tr_norm[i] = (seq - seq[0]) / (seq[0] + 1e-8)

        for i in range(X_te.shape[0]):
            seq = X_te[i]
            X_te_norm[i] = (seq - seq[0]) / (seq[0] + 1e-8)

        return X_tr_norm, X_te_norm

    # Test all methods
    results = []
    results.append(("Global MinMax", *test_method(X_train, X_test, y_train, y_test, "Global MinMax", method1_global_minmax)))
    results.append(("Per-Sequence MinMax", *test_method(X_train, X_test, y_train, y_test, "Per-Sequence MinMax", method2_per_sequence_minmax)))
    results.append(("StandardScaler", *test_method(X_train, X_test, y_train, y_test, "StandardScaler (Z-score)", method3_standardscaler)))
    results.append(("Returns-based", *test_method(X_train, X_test, y_train, y_test, "Returns-based (pct change)", method4_returns_based)))

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, diverse_probs, diverse_preds in results:
        status = "✅" if (diverse_probs and diverse_preds) else "❌"
        print(f"{status} {name}: diverse_probs={diverse_probs}, diverse_preds={diverse_preds}")

if __name__ == '__main__':
    main()
