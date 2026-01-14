#!/usr/bin/env python3
"""
Debug script to investigate sequence model predictions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.features import FeatureEngineer
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
    print(f"   Collected {len(data)} records")

    # Preprocess
    print("\n2. Preprocessing...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(data)
    print(f"   Preprocessed: {len(data)} records")

    # Feature engineering
    print("\n3. Feature engineering...")
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)
    print(f"   Features: {len(data.columns)} columns")

    # Create target
    print("\n4. Creating target...")
    future_return = data['close'].shift(-1) / data['close'] - 1
    data['profitable_trade'] = (future_return > 0).astype(int)
    data = data.dropna()
    print(f"   Target distribution: {np.bincount(data['profitable_trade'].values.astype(int))}")
    print(f"   Positive rate: {data['profitable_trade'].mean():.2%}")

    # Split data
    print("\n5. Splitting data...")
    train_size = 365
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:train_size+60].copy()
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")

    # Create sequences
    print("\n6. Creating sequences...")
    seq_gen = SequenceGenerator(sequence_length=30, target_horizon=1)

    X_train_seq, y_train_seq, _ = seq_gen.create_sequences(
        train_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )
    print(f"   Train sequences: {X_train_seq.shape}, targets: {y_train_seq.shape}")
    print(f"   Train target distribution: {np.bincount(y_train_seq.astype(int))}")

    X_test_seq, y_test_seq, _ = seq_gen.create_sequences(
        test_data,
        ['open', 'high', 'low', 'close', 'volume'],
        'profitable_trade'
    )
    print(f"   Test sequences: {X_test_seq.shape}, targets: {y_test_seq.shape}")
    print(f"   Test target distribution: {np.bincount(y_test_seq.astype(int))}")

    # Train LSTM
    print("\n7. Training LSTM...")
    lstm = SequenceLSTMModel(
        model_type='classification',
        params={
            'units': 64,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32
        }
    )
    lstm.fit(X_train_seq, y_train_seq)

    # Predict with LSTM
    print("\n8. LSTM Predictions...")
    lstm_pred = lstm.predict(X_test_seq)
    lstm_proba = lstm.predict_proba(X_test_seq)

    print(f"   Predictions shape: {lstm_pred.shape}")
    print(f"   Predictions type: {lstm_pred.dtype}")
    print(f"   Prediction distribution: {np.bincount(lstm_pred.astype(int))}")
    print(f"   Unique predictions: {np.unique(lstm_pred)}")
    print(f"   First 10 predictions: {lstm_pred[:10]}")
    print(f"\n   Probabilities shape: {lstm_proba.shape}")
    print(f"   First 10 probabilities (class 0, class 1):")
    for i in range(min(10, len(lstm_proba))):
        print(f"      Sample {i}: [{lstm_proba[i][0]:.4f}, {lstm_proba[i][1]:.4f}] → pred={lstm_pred[i]}")

    # Check if predictions are all the same
    if len(np.unique(lstm_pred)) == 1:
        print(f"\n   ⚠️  WARNING: All predictions are {lstm_pred[0]}!")
        print(f"   This explains the 0% returns!")

    # Train Transformer
    print("\n9. Training Transformer...")
    transformer = SequenceTransformerModel(
        model_type='classification',
        params={
            'd_model': 64,
            'n_heads': 4,
            'ff_dim': 128,
            'dropout': 0.2,
            'epochs': 20,
            'batch_size': 32
        }
    )
    transformer.fit(X_train_seq, y_train_seq)

    # Predict with Transformer
    print("\n10. Transformer Predictions...")
    transformer_pred = transformer.predict(X_test_seq)
    transformer_proba = transformer.predict_proba(X_test_seq)

    print(f"   Predictions shape: {transformer_pred.shape}")
    print(f"   Prediction distribution: {np.bincount(transformer_pred.astype(int))}")
    print(f"   Unique predictions: {np.unique(transformer_pred)}")
    print(f"   First 10 predictions: {transformer_pred[:10]}")
    print(f"\n   First 10 probabilities (class 0, class 1):")
    for i in range(min(10, len(transformer_proba))):
        print(f"      Sample {i}: [{transformer_proba[i][0]:.4f}, {transformer_proba[i][1]:.4f}] → pred={transformer_pred[i]}")

    if len(np.unique(transformer_pred)) == 1:
        print(f"\n   ⚠️  WARNING: All predictions are {transformer_pred[0]}!")

    # Compare predictions
    print("\n11. Comparing Models...")
    same_preds = np.sum(lstm_pred == transformer_pred)
    print(f"   Same predictions: {same_preds}/{len(lstm_pred)} ({same_preds/len(lstm_pred)*100:.1f}%)")

    if same_preds == len(lstm_pred):
        print(f"   ⚠️  Both models predict identically!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Train target balance: {np.bincount(y_train_seq.astype(int))}")
    print(f"Test target balance: {np.bincount(y_test_seq.astype(int))}")
    print(f"LSTM predictions: {np.bincount(lstm_pred.astype(int))}")
    print(f"Transformer predictions: {np.bincount(transformer_pred.astype(int))}")
    print(f"Identical predictions: {same_preds}/{len(lstm_pred)}")

    # Calculate accuracy
    lstm_acc = np.mean(lstm_pred == y_test_seq)
    transformer_acc = np.mean(transformer_pred == y_test_seq)
    print(f"\nLSTM accuracy: {lstm_acc:.2%}")
    print(f"Transformer accuracy: {transformer_acc:.2%}")

if __name__ == '__main__':
    main()
