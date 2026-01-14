#!/usr/bin/env python3
"""Test StandardScaler normalization (same as traditional models)."""
import sys
sys.path.insert(0, '/home/user/TradingBot-8888')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from trading_bot.data.sequences import SequenceGenerator
from trading_bot.models.sequence_models import SequenceLSTMModel

print("Testing StandardScaler (Z-score) normalization...")

# Get data
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

# Split
train_data = data.iloc[:320].copy()
test_data = data.iloc[320:380].copy()  # Need at least 30 samples for sequences

# Create sequences
seq_gen = SequenceGenerator(sequence_length=30, target_horizon=1)
X_train, y_train, _ = seq_gen.create_sequences(train_data, ['open', 'high', 'low', 'close', 'volume'], 'profitable_trade')
X_test, y_test, _ = seq_gen.create_sequences(test_data, ['open', 'high', 'low', 'close', 'volume'], 'profitable_trade')

print(f"\nOriginal ranges:")
print(f"  Train: [{X_train.min():.0f}, {X_train.max():.0f}]")
print(f"  Test: [{X_test.min():.0f}, {X_test.max():.0f}]")

# Normalize with StandardScaler (same as traditional models)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
X_test_norm = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)

print(f"\nNormalized (StandardScaler):")
print(f"  Train: mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}")
print(f"  Test: mean={X_test_norm.mean():.4f}, std={X_test_norm.std():.4f}")
print(f"  Train range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
print(f"  Test range: [{X_test_norm.min():.2f}, {X_test_norm.max():.2f}]")

# Train LSTM
print(f"\nTraining LSTM with StandardScaler normalization...")
lstm = SequenceLSTMModel(
    model_type='classification',
    params={'units': 32, 'dropout': 0.3, 'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32, 'patience': 5}
)
lstm.fit(X_train_norm, y_train)

# Predict
predictions = lstm.predict(X_test_norm)
probabilities = lstm.predict_proba(X_test_norm)

print(f"\nPredictions:")
print(f"  Distribution: {np.bincount(predictions.astype(int))}")
print(f"  Unique: {len(np.unique(predictions))}")

prob_std = probabilities.std(axis=0)
print(f"  Probability std: [class 0: {prob_std[0]:.6f}, class 1: {prob_std[1]:.6f}]")

print(f"\nFirst 10 predictions:")
for i in range(min(10, len(predictions))):
    print(f"  Sample {i}: [{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}] → {predictions[i]} (actual={int(y_test[i])})")

if prob_std[0] > 0.01:
    print(f"\n✅ SUCCESS: Model produces diverse predictions!")
    print(f"   StandardScaler works much better than MinMaxScaler")
else:
    print(f"\n❌ Still degenerate: std={prob_std[0]:.6f}")

accuracy = np.mean(predictions == y_test)
print(f"\nAccuracy: {accuracy:.2%}")
