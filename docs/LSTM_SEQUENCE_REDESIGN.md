# LSTM/Transformer Sequence Redesign Plan

## Executive Summary

**Goal**: Properly implement LSTM/Transformer with raw OHLCV sequences instead of the current broken 1-timestep approach.

**Key Decision**: Use **raw sequences** (no technical indicators) to let the network learn temporal patterns naturally.

---

## 1. Architecture Design

### Core Concept

**Current (BROKEN)**:
```
Input: [62 features] → Reshape to (1, 62) → LSTM → Output
└─ Only 1 timestep, equivalent to Dense network
```

**New (PROPER)**:
```
Input: [60 days × 5 OHLCV] → LSTM layers → Output
└─ 60 timesteps, learns temporal dependencies
```

### Data Structure

```python
# Raw OHLCV sequence
Sequence shape: (batch_size, sequence_length, 5)
  ├─ sequence_length: 30, 60, 90, or 180 days
  └─ 5 features: [open, high, low, close, volume]

# Optional: Add normalized features
Enhanced shape: (batch_size, sequence_length, 8)
  ├─ 5 OHLCV features
  ├─ returns (close.pct_change())
  ├─ high/low ratio
  └─ volume change
```

---

## 2. Sequence Length Evaluation Plan

### Factors to Consider

| Sequence Length | Pros | Cons | Best For |
|----------------|------|------|----------|
| **30 days** | Fast training, less overfitting | Misses longer trends | 1-day, 7-day horizons |
| **60 days** | Good balance, ~2 months context | Medium compute | 7-day, 14-day horizons |
| **90 days** | Captures quarterly patterns | Slower training | 14-day, 28-day horizons |
| **180 days** | Long-term trends, 6-month context | Heavy compute, overfitting risk | 28-day, 60-day horizons |

### Recommended Testing Matrix

```python
experiments = {
    '1day_horizon': {
        'sequence_lengths': [30, 60],
        'reason': 'Short-term prediction needs recent context'
    },
    '7day_horizon': {
        'sequence_lengths': [30, 60, 90],
        'reason': 'Week ahead benefits from 1-3 months context'
    },
    '14day_horizon': {
        'sequence_lengths': [60, 90],
        'reason': '2 weeks ahead needs 2-3 months context'
    },
    '28day_horizon': {
        'sequence_lengths': [90, 180],
        'reason': 'Month ahead needs quarterly context'
    },
    '60day_horizon': {
        'sequence_lengths': [90, 180],
        'reason': '2 months ahead needs 3-6 months context'
    }
}
```

### Memory & Compute Considerations

```python
# Memory usage estimate
def estimate_memory(batch_size, seq_length, num_samples):
    # Float32 = 4 bytes
    input_size = batch_size * seq_length * 5 * 4
    
    # LSTM hidden states (approximate)
    hidden_size = batch_size * seq_length * 128 * 4 * 2  # Forward + backward
    
    total_mb = (input_size + hidden_size) / (1024 * 1024)
    
    return total_mb

# Example:
# 32 batch × 180 seq × 5 features = 27 MB input
# With hidden states: ~500 MB per batch
# Feasible on GPU, challenging on CPU
```

**Recommendation**: 
- Start with **60 days** (good balance)
- Test **30 days** for 1-day horizon
- Test **90 days** for 28-day, 60-day horizons
- **Skip 180 days** initially (too heavy, use if 90 doesn't work)

---

## 3. Proposed Architecture

### Option A: Pure LSTM (Simpler, Faster)

```python
class SequenceLSTM(nn.Module):
    def __init__(self, sequence_length=60, input_features=5):
        super().__init__()
        
        # Input normalization
        self.batch_norm = nn.BatchNorm1d(input_features)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # Look forward AND backward in time
        )
        
        self.lstm2 = nn.LSTM(
            input_size=256,  # 128 * 2 (bidirectional)
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism (focus on important timesteps)
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Normalize each feature across sequence
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Attention (learn which timesteps matter most)
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        
        # Take last timestep output
        last_hidden = attn_out[:, -1, :]
        
        # Dense layers
        out = F.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
```

**Why this architecture?**
- **Bidirectional LSTM**: Sees future context during training (useful for pattern learning)
- **Attention**: Focuses on important time periods (e.g., volatility spikes)
- **BatchNorm**: Handles varying scales of OHLCV
- **Dropout**: Prevents overfitting

---

### Option B: Transformer (More Advanced)

```python
class SequenceTransformer(nn.Module):
    def __init__(self, sequence_length=60, input_features=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.d_model = 128
        
        # Input projection
        self.input_projection = nn.Linear(input_features, self.d_model)
        
        # Positional encoding (tells model about time order)
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output layers
        self.fc1 = nn.Linear(self.d_model, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        
    def _create_positional_encoding(self):
        pe = torch.zeros(self.sequence_length, self.d_model)
        position = torch.arange(0, self.sequence_length).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                             -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Take last timestep or pool
        last_hidden = transformer_out[:, -1, :]
        
        # Output
        out = F.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
```

**Why Transformer?**
- **Self-attention**: Can look at ANY past timestep directly (not sequential like LSTM)
- **Parallelizable**: Faster training on GPU
- **Long-range dependencies**: Better for 90+ day sequences
- **State-of-the-art**: Used in GPT, BERT, etc.

---

## 4. Data Pipeline Changes

### Sequence Creation

```python
class SequenceDataset(torch.utils.data.Dataset):
    """Create overlapping sequences for LSTM/Transformer training"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int, 
                 prediction_horizon: int, target_col: str):
        """
        Args:
            data: DataFrame with OHLCV columns
            sequence_length: Number of days to look back (30, 60, 90, 180)
            prediction_horizon: Days ahead to predict (1, 7, 14, 28, 60)
            target_col: Target column name
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Extract OHLCV
        self.ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Extract target (future return)
        self.targets = data[target_col].values
        
        # Valid indices (need enough history + future)
        self.valid_indices = range(
            sequence_length, 
            len(data) - prediction_horizon
        )
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get sequence (past 60 days)
        sequence_start = actual_idx - self.sequence_length
        sequence_end = actual_idx
        sequence = self.ohlcv[sequence_start:sequence_end]
        
        # Get target (future return)
        target = self.targets[actual_idx]
        
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


# Normalization (critical!)
class SequenceNormalizer:
    """Normalize sequences while preserving temporal relationships"""
    
    def __init__(self):
        self.price_mean = None
        self.price_std = None
        self.volume_mean = None
        self.volume_std = None
    
    def fit(self, sequences):
        """Fit on training sequences"""
        # Flatten all sequences
        all_prices = sequences[:, :, :4].reshape(-1, 4)  # OHLC
        all_volumes = sequences[:, :, 4].reshape(-1)
        
        self.price_mean = all_prices.mean(axis=0)
        self.price_std = all_prices.std(axis=0)
        self.volume_mean = all_volumes.mean()
        self.volume_std = all_volumes.std()
    
    def transform(self, sequences):
        """Normalize sequences"""
        normalized = sequences.copy()
        
        # Normalize OHLC
        normalized[:, :, :4] = (sequences[:, :, :4] - self.price_mean) / self.price_std
        
        # Normalize volume
        normalized[:, :, 4] = (sequences[:, :, 4] - self.volume_mean) / self.volume_std
        
        return normalized
```

---

## 5. Walk-Forward Integration

### Modified Training Flow

```python
class SequenceWalkForwardTester(WalkForwardTester):
    """Walk-forward testing for sequence models"""
    
    def train_sequence_models_for_window(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        sequence_length: int,
        window_idx: int
    ):
        """Train LSTM/Transformer with sequences"""
        
        # Create sequence dataset
        dataset = SequenceDataset(
            data=train_data,
            sequence_length=sequence_length,
            prediction_horizon=self.prediction_horizon,
            target_col=target_col
        )
        
        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        # Initialize models
        models = {}
        
        # LSTM model
        lstm_model = SequenceLSTM(
            sequence_length=sequence_length,
            input_features=5
        )
        lstm_model = self.train_sequence_model(lstm_model, train_loader, 'lstm')
        models['lstm'] = lstm_model
        
        # Transformer model
        transformer_model = SequenceTransformer(
            sequence_length=sequence_length,
            input_features=5
        )
        transformer_model = self.train_sequence_model(transformer_model, train_loader, 'transformer')
        models['transformer'] = transformer_model
        
        return models
    
    def train_sequence_model(self, model, train_loader, name):
        """Train a single sequence model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  {name} early stopping at epoch {epoch}")
                break
        
        return model
```

---

## 6. Experiment Protocol

### Phase 1: Baseline Sequence Experiments (Week 1-2)

**Goal**: Find optimal sequence length per horizon

```bash
# Test each horizon with 2-3 sequence lengths
python scripts/sequence_walk_forward_test.py \
  --horizon 1 --sequence-length 30 --days 3650

python scripts/sequence_walk_forward_test.py \
  --horizon 1 --sequence-length 60 --days 3650

python scripts/sequence_walk_forward_test.py \
  --horizon 7 --sequence-length 30 --days 3650

python scripts/sequence_walk_forward_test.py \
  --horizon 7 --sequence-length 60 --days 3650

# ... repeat for all horizons
```

**Success Criteria**:
- Beat tree-based models (XGBoost, LightGBM) by >5pp
- Beat Buy & Hold by >10pp
- Stable across multiple walk-forward windows

---

### Phase 2: Architecture Comparison (Week 3)

**Goal**: LSTM vs Transformer

```python
# Compare architectures with best sequence length from Phase 1
results = {
    'lstm_bidirectional': test_model(LSTMBidirectional),
    'lstm_attention': test_model(LSTMWithAttention),
    'transformer_4layers': test_model(Transformer4Layers),
    'transformer_8layers': test_model(Transformer8Layers),
}
```

**Success Criteria**:
- One architecture clearly dominates
- Training time < 30 minutes per window
- Inference time < 1 second per prediction

---

### Phase 3: Hybrid Approach (Week 4)

**Goal**: Combine sequences with selected features

```python
# Add top 10 features to sequence model
enhanced_features = [
    'rsi_14',
    'macd',
    'volume_sma_20',
    'bb_width',
    'atr',
    'adx',
    'obv',
    'fear_greed_index',
    'price_ma_distance_20',
    'volatility_30d'
]

# Hybrid architecture:
# Path 1: Sequences → LSTM
# Path 2: Features → Dense
# Concatenate → Output
```

---

## 7. Expected Outcomes by Sequence Length

### 30-Day Sequences

**Best for**: 1-day, 7-day horizons

**Pros**:
- Fast training (~5-10 min per window)
- Less overfitting
- Captures recent volatility

**Cons**:
- Misses longer-term trends
- May not help 28-day, 60-day predictions

**Expected improvement over current**:
- 1-day horizon: +20-40pp (currently -21% LSTM, target +20%)
- 7-day horizon: +30-50pp (currently -22% LSTM, target +30%)

---

### 60-Day Sequences (RECOMMENDED)

**Best for**: All horizons (good default)

**Pros**:
- Captures ~2 months of patterns
- Good balance of context vs compute
- Training time acceptable (~10-15 min per window)

**Cons**:
- Moderate memory usage
- May need GPU for large batches

**Expected improvement over current**:
- 1-day: +15-30pp
- 7-day: +40-60pp (should beat current tree models)
- 14-day: +20-40pp
- 28-day: +30-50pp
- 60-day: +20-30pp

**This is the sweet spot - start here.**

---

### 90-Day Sequences

**Best for**: 14-day, 28-day, 60-day horizons

**Pros**:
- Captures quarterly patterns
- Better for longer-term predictions
- Can detect regime changes

**Cons**:
- Slower training (~20-30 min per window)
- Higher memory usage (need GPU)
- Risk of overfitting

**Expected improvement**:
- 28-day: +40-60pp (currently +88% LSTM, target +120%+)
- 60-day: +30-50pp (currently +112% LSTM, target +140%+)

---

### 180-Day Sequences

**Best for**: 60-day horizon only

**Pros**:
- Maximum context (6 months)
- Captures full bull/bear mini-cycles
- Can learn macro trends

**Cons**:
- Very slow training (~45-60 min per window)
- Heavy memory usage (requires GPU with 8GB+ VRAM)
- High overfitting risk
- Need MORE historical data (10+ years)

**Expected improvement**:
- 60-day: +40-70pp (if it works)

**Recommendation**: Skip unless 90-day fails for 60-day horizon

---

## 8. Implementation Checklist

### Week 1: Infrastructure
- [ ] Create `SequenceDataset` class
- [ ] Create `SequenceNormalizer` class
- [ ] Modify `walk_forward_test.py` for sequences
- [ ] Add PyTorch dependency
- [ ] Test on small dataset (100 samples)

### Week 2: LSTM Implementation
- [ ] Implement `SequenceLSTM` architecture
- [ ] Test 30-day, 60-day sequences
- [ ] Run on 1-day, 7-day horizons
- [ ] Compare vs current LSTM
- [ ] Analyze attention weights

### Week 3: Transformer Implementation
- [ ] Implement `SequenceTransformer` architecture
- [ ] Test same sequence lengths
- [ ] Compare LSTM vs Transformer
- [ ] Select best architecture

### Week 4: Full Evaluation
- [ ] Run all 5 horizons
- [ ] Test optimal sequence length per horizon
- [ ] Walk-forward on 10 years data
- [ ] Compare vs tree models
- [ ] Document results

---

## 9. Risk Mitigation

### Risk 1: Overfitting
**Symptom**: Training loss ↓, validation loss ↑

**Mitigations**:
- Increase dropout (0.3 → 0.5)
- Add L2 regularization
- Use smaller models (64 hidden units instead of 128)
- Shorter sequences (60 → 30 days)

### Risk 2: Training Too Slow
**Symptom**: >1 hour per window

**Mitigations**:
- Use GPU (essential for Transformer)
- Reduce sequence length
- Smaller batch size (32 → 16)
- Fewer LSTM/Transformer layers

### Risk 3: Still Underperforms Tree Models
**Symptom**: LSTM/Transformer < XGBoost after redesign

**Decision**: Remove them entirely, focus on tree models

**Why**: Not all problems need deep learning - if trees work better, use trees!

---

## 10. Success Criteria

### Minimum Viable Success
- [ ] Beat current broken LSTM/Transformer (easy bar)
- [ ] Match tree models (XGBoost, LightGBM)
- [ ] Beat Buy & Hold on 3/5 horizons

### Target Success
- [ ] Beat tree models by >10pp on 3/5 horizons
- [ ] Beat Buy & Hold on 4/5 horizons
- [ ] Stable across walk-forward windows

### Exceptional Success
- [ ] Beat all baselines on all horizons
- [ ] 60-day sequences work for all horizons
- [ ] Transformer > LSTM (cutting edge validation)

---

## 11. Final Recommendations

### Start Here (Phase 1, Week 1-2)

**Priority 1: 60-Day LSTM**
```bash
# Test 60-day sequences with LSTM on 1-day horizon
python scripts/sequence_walk_forward_test.py \
  --model lstm \
  --horizon 1 \
  --sequence-length 60 \
  --days 3650 \
  --architecture bidirectional_attention

# Expected: +15-30pp improvement over current LSTM
```

**Priority 2: 30-Day LSTM (if 60 too slow)**
```bash
# Faster alternative
python scripts/sequence_walk_forward_test.py \
  --model lstm \
  --horizon 1 \
  --sequence-length 30 \
  --days 3650

# Expected: +10-20pp improvement
```

**Priority 3: Multi-Horizon Sweep**
```bash
# Test optimal lengths per horizon
for horizon in 1 7 14 28 60; do
    for seqlen in 30 60 90; do
        python scripts/sequence_walk_forward_test.py \
          --model lstm \
          --horizon $horizon \
          --sequence-length $seqlen \
          --days 3650
    done
done
```

### Decision Tree

```
Start: Implement 60-day LSTM with attention
  ├─ Works (beats tree models)? 
  │   ├─ YES → Test Transformer
  │   │   ├─ Transformer better? Use Transformer
  │   │   └─ LSTM better? Use LSTM
  │   └─ NO → Try 30-day sequences
  │       ├─ Works? Use 30-day
  │       └─ NO → REMOVE sequences entirely, stick with tree models
  │
  └─ Too slow?
      └─ Try 30-day sequences first
```

---

## 12. Estimated Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| **1** | Infrastructure + Data Pipeline | SequenceDataset, normalizer working |
| **2** | LSTM Implementation | 30-day, 60-day LSTM trained on 1-day, 7-day |
| **3** | Transformer + Comparison | Best architecture selected |
| **4** | Full Evaluation | All 5 horizons tested, report generated |

**Total**: 4 weeks to complete redesign and evaluation

---

## Conclusion

**Recommendation**: Start with **60-day sequences + bidirectional LSTM + attention**

**Why 60 days?**
- Best balance of context and compute
- Works for most horizons
- Not too slow, not too short

**Why LSTM over Transformer?**
- Simpler to implement
- Faster training
- Proven track record in finance
- Try Transformer in Week 3 if LSTM works

**Why Attention?**
- Learns which time periods matter most
- Interpretable (can visualize attention weights)
- Small overhead, big benefit

**Next Step**: Create `scripts/sequence_walk_forward_test.py` with the architecture above.
