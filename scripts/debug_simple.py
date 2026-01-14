#!/usr/bin/env python3
"""
Simple debug script to investigate sequence model predictions.
"""

import sys
sys.path.insert(0, '/home/user/TradingBot-8888')

import numpy as np
import pandas as pd
import yfinance as yf

from trading_bot.data.sequences import SequenceGenerator
from trading_bot.models.sequence_models import SequenceLSTMModel, SequenceTransformerModel

def main():
    print("=" * 80)
    print("SEQUENCE MODEL PREDICTION DEBUG")
    print("=" * 80)

    # Collect data
    print("\n1. Collecting data...")
    ticker = yf.Ticker('BTC-USD')
    data = ticker.history(period='400d', interval='1d')
    data = data.reset_index()
    data.columns = [c.lower() for c in data.columns]
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    data = data.set_index('date')
    print(f"   Collected {len(data)} records")

    # Create target
    print("\n2. Creating target...")
    future_return = data['close'].shift(-1) / data['close'] - 1
    data['profitable_trade'] = (future_return > 0).astype(int)
    data = data.dropna()
    print(f"   Target distribution: {np.bincount(data['profitable_trade'].values.astype(int))}")
    print(f"   Positive rate: {data['profitable_trade'].mean():.2%}")

    # Split data
    print("\n3. Splitting data...")
    train_size = 365
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:train_size+60].copy()
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"   Train target: {np.bincount(train_data['profitable_trade'].values.astype(int))}")
    print(f"   Test target: {np.bincount(test_data['profitable_trade'].values.astype(int))}")

    # Create sequences
    print("\n4. Creating sequences...")
    seq_gen = SequenceGenerator(sequence_length=30, target_horizon=1)

    X_train_seq, y_train_seq, _ = seq_gen.create_sequences(
        train_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )
    print(f"   Train sequences: {X_train_seq.shape}, targets: {y_train_seq.shape}")
    print(f"   Train target distribution: {np.bincount(y_train_seq.astype(int))}")
    print(f"   Train positive rate: {y_train_seq.mean():.2%}")

    X_test_seq, y_test_seq, _ = seq_gen.create_sequences(
        test_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )
    print(f"   Test sequences: {X_test_seq.shape}, targets: {y_test_seq.shape}")
    print(f"   Test target distribution: {np.bincount(y_test_seq.astype(int))}")
    print(f"   Test positive rate: {y_test_seq.mean():.2%}")

    # Train LSTM
    print("\n5. Training LSTM...")
    lstm = SequenceLSTMModel(
        model_type='classification',
        params={
            'units': 64,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 5,  # Fewer epochs for faster debug
            'batch_size': 32,
            'patience': 3
        }
    )
    lstm.fit(X_train_seq, y_train_seq)

    # Predict with LSTM
    print("\n6. LSTM Predictions...")
    lstm_pred = lstm.predict(X_test_seq)
    lstm_proba = lstm.predict_proba(X_test_seq)

    print(f"   Predictions shape: {lstm_pred.shape}")
    print(f"   Predictions type: {lstm_pred.dtype}")
    print(f"   Prediction distribution: {np.bincount(lstm_pred.astype(int))}")
    print(f"   Unique predictions: {np.unique(lstm_pred)}")
    print(f"   First 20 predictions: {lstm_pred[:20]}")
    print(f"\n   Probabilities shape: {lstm_proba.shape}")
    print(f"   First 10 probabilities (class 0, class 1):")
    for i in range(min(10, len(lstm_proba))):
        print(f"      Sample {i}: [{lstm_proba[i][0]:.4f}, {lstm_proba[i][1]:.4f}] → pred={lstm_pred[i]} (actual={int(y_test_seq[i])})")

    # Check if predictions are all the same
    if len(np.unique(lstm_pred)) == 1:
        print(f"\n   ⚠️  WARNING: All predictions are {lstm_pred[0]}!")
        print(f"   This explains the 0% returns!")
        print(f"\n   Checking probabilities...")
        print(f"   Min prob class 0: {lstm_proba[:, 0].min():.4f}")
        print(f"   Max prob class 0: {lstm_proba[:, 0].max():.4f}")
        print(f"   Mean prob class 0: {lstm_proba[:, 0].mean():.4f}")
        print(f"   Std prob class 0: {lstm_proba[:, 0].std():.4f}")

    # Calculate accuracy
    lstm_acc = np.mean(lstm_pred == y_test_seq)
    print(f"\n   LSTM accuracy: {lstm_acc:.2%}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Train target balance: {np.bincount(y_train_seq.astype(int))} ({y_train_seq.mean():.1%} positive)")
    print(f"Test target balance: {np.bincount(y_test_seq.astype(int))} ({y_test_seq.mean():.1%} positive)")
    print(f"LSTM predictions: {np.bincount(lstm_pred.astype(int))}")
    print(f"LSTM accuracy: {lstm_acc:.2%}")

    if len(np.unique(lstm_pred)) == 1:
        print(f"\n⚠️  LSTM predicts constant class {lstm_pred[0]}")
        print(f"   This is a degenerate model that needs investigation!")

if __name__ == '__main__':
    main()
