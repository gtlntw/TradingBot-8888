#!/usr/bin/env python3
"""
Visualize per-window performance across all horizons and models.

Creates comprehensive charts showing:
- Per-window returns for each model
- Consistency across windows
- Comparison by horizon
"""

import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_results():
    """Load all JSON results files."""
    horizons = [1, 7, 14, 28, 60]
    all_data = {}

    for horizon in horizons:
        pattern = f'experiments/walk_forward_enhanced/enhanced_wf_{horizon}day_*.json'
        files = glob.glob(pattern)

        if files:
            with open(files[0], 'r') as f:
                all_data[horizon] = json.load(f)

    return all_data

def extract_window_returns(data, horizon):
    """Extract per-window returns for all models."""
    window_returns = {}

    for window in data['windows']:
        window_num = window['window']

        for model_name, model_result in window['results'].items():
            if model_result and 'total_return' in model_result:
                if model_name not in window_returns:
                    window_returns[model_name] = {}
                window_returns[model_name][window_num] = model_result['total_return'] * 100

    return window_returns

def plot_all_horizons_heatmap(all_data):
    """Create heatmap showing returns across windows and horizons."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Per-Window Returns by Horizon and Model (% Return)', fontsize=16, fontweight='bold')

    horizons = [1, 7, 14, 28, 60]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // 3, idx % 3]

        if horizon not in all_data:
            ax.text(0.5, 0.5, f'No data for {horizon}-day', ha='center', va='center')
            ax.set_title(f'{horizon}-Day Horizon')
            continue

        window_returns = extract_window_returns(all_data[horizon], horizon)

        # Create DataFrame for heatmap
        models = ['random_forest', 'xgboost', 'lightgbm', 'lstm', 'transformer',
                  'ensemble_traditional', 'lstm_60day', 'transformer_60day', 'buy_and_hold']

        df_data = []
        for model in models:
            if model in window_returns:
                row = [window_returns[model].get(i, np.nan) for i in range(1, 11)]
                df_data.append(row)
            else:
                df_data.append([np.nan] * 10)

        df = pd.DataFrame(df_data, index=models, columns=[f'W{i}' for i in range(1, 11)])

        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'}, ax=ax, vmin=-30, vmax=30)
        ax.set_title(f'{horizon}-Day Horizon', fontsize=12, fontweight='bold')
        ax.set_xlabel('Window')
        ax.set_ylabel('Model')

    # Hide the 6th subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/walk_forward_enhanced/heatmap_all_horizons.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heatmap_all_horizons.png")
    plt.close()

def plot_top_models_by_horizon(all_data):
    """Plot line charts for top 3 models per horizon."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Top 3 Models Performance Across Windows', fontsize=16, fontweight='bold')

    horizons = [1, 7, 14, 28, 60]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // 3, idx % 3]

        if horizon not in all_data:
            continue

        window_returns = extract_window_returns(all_data[horizon], horizon)

        # Calculate mean return for each model
        model_means = {}
        for model, returns in window_returns.items():
            if model != 'buy_and_hold':
                model_means[model] = np.mean(list(returns.values()))

        # Get top 3 models
        top_3 = sorted(model_means.items(), key=lambda x: x[1], reverse=True)[:3]

        # Plot top 3 + buy_and_hold
        windows = list(range(1, 11))

        for model_name, _ in top_3:
            returns = [window_returns[model_name].get(i, np.nan) for i in windows]
            ax.plot(windows, returns, marker='o', linewidth=2, label=model_name, markersize=6)

        # Add buy and hold
        if 'buy_and_hold' in window_returns:
            bh_returns = [window_returns['buy_and_hold'].get(i, np.nan) for i in windows]
            ax.plot(windows, bh_returns, marker='s', linewidth=2, label='buy_and_hold',
                   linestyle='--', color='black', markersize=6, alpha=0.7)

        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax.set_title(f'{horizon}-Day Horizon', fontsize=12, fontweight='bold')
        ax.set_xlabel('Window')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(windows)

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/walk_forward_enhanced/line_chart_top_models.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: line_chart_top_models.png")
    plt.close()

def plot_model_consistency(all_data):
    """Plot box plots showing return distribution across windows."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Return Distribution Across 10 Windows (Box Plots)', fontsize=16, fontweight='bold')

    horizons = [1, 7, 14, 28, 60]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // 3, idx % 3]

        if horizon not in all_data:
            continue

        window_returns = extract_window_returns(all_data[horizon], horizon)

        # Prepare data for box plot
        models_to_plot = ['random_forest', 'xgboost', 'lightgbm', 'lstm', 'transformer',
                         'ensemble_traditional', 'lstm_60day', 'transformer_60day', 'buy_and_hold']

        data_for_plot = []
        labels = []

        for model in models_to_plot:
            if model in window_returns:
                returns = list(window_returns[model].values())
                if returns:
                    data_for_plot.append(returns)
                    # Shorten labels
                    label = model.replace('_traditional', '').replace('_60day', '_seq')
                    labels.append(label[:12])

        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(range(len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax.set_title(f'{horizon}-Day Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/walk_forward_enhanced/boxplot_consistency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: boxplot_consistency.png")
    plt.close()

def plot_sequence_vs_traditional(all_data):
    """Compare sequence models vs traditional models."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Sequence Models vs Traditional Models (Per Window)', fontsize=16, fontweight='bold')

    horizons = [1, 7, 14, 28, 60]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // 3, idx % 3]

        if horizon not in all_data:
            continue

        window_returns = extract_window_returns(all_data[horizon], horizon)
        windows = list(range(1, 11))

        # Best traditional model
        trad_models = ['random_forest', 'xgboost', 'lightgbm', 'lstm', 'transformer']
        trad_means = {m: np.mean(list(window_returns.get(m, {}).values() or [0]))
                     for m in trad_models if m in window_returns}
        if trad_means:
            best_trad = max(trad_means, key=trad_means.get)
            trad_returns = [window_returns[best_trad].get(i, np.nan) for i in windows]
            ax.plot(windows, trad_returns, marker='o', linewidth=2.5,
                   label=f'{best_trad} (best trad)', markersize=8)

        # Sequence models
        seq_models = ['lstm_60day', 'transformer_60day']
        for seq in seq_models:
            if seq in window_returns:
                seq_returns = [window_returns[seq].get(i, np.nan) for i in windows]
                ax.plot(windows, seq_returns, marker='s', linewidth=2.5,
                       label=seq, markersize=8)

        # Buy and hold
        if 'buy_and_hold' in window_returns:
            bh_returns = [window_returns['buy_and_hold'].get(i, np.nan) for i in windows]
            ax.plot(windows, bh_returns, marker='d', linewidth=2,
                   label='buy_and_hold', linestyle='--', color='black', markersize=6)

        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax.set_title(f'{horizon}-Day Horizon', fontsize=12, fontweight='bold')
        ax.set_xlabel('Window')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(windows)

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/walk_forward_enhanced/sequence_vs_traditional.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sequence_vs_traditional.png")
    plt.close()

def plot_win_loss_analysis(all_data):
    """Plot win/loss patterns across windows."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Win/Loss Pattern Across Windows (Green=Win, Red=Loss)', fontsize=16, fontweight='bold')

    horizons = [1, 7, 14, 28, 60]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx // 3, idx % 3]

        if horizon not in all_data:
            continue

        window_returns = extract_window_returns(all_data[horizon], horizon)

        # Top 5 models
        model_means = {m: np.mean(list(r.values()))
                      for m, r in window_returns.items() if m != 'buy_and_hold'}
        top_5 = [m for m, _ in sorted(model_means.items(), key=lambda x: x[1], reverse=True)[:5]]

        # Create win/loss matrix
        windows = list(range(1, 11))
        for i, model in enumerate(top_5):
            returns = [window_returns[model].get(w, 0) for w in windows]
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax.scatter([i] * len(windows), windows, c=colors, s=200, alpha=0.6, marker='s')

        ax.set_yticks(windows)
        ax.set_xticks(range(len(top_5)))
        ax.set_xticklabels([m[:15] for m in top_5], rotation=45, ha='right')
        ax.set_title(f'{horizon}-Day Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel('Window')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('experiments/walk_forward_enhanced/win_loss_pattern.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: win_loss_pattern.png")
    plt.close()

def main():
    print("\n" + "="*80)
    print("VISUALIZING WINDOW PERFORMANCE")
    print("="*80 + "\n")

    # Load data
    print("Loading results...")
    all_data = load_results()
    print(f"✓ Loaded data for {len(all_data)} horizons\n")

    # Create output directory
    Path('experiments/walk_forward_enhanced').mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("Generating visualizations...")

    plot_all_horizons_heatmap(all_data)
    plot_top_models_by_horizon(all_data)
    plot_model_consistency(all_data)
    plot_sequence_vs_traditional(all_data)
    plot_win_loss_analysis(all_data)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files in experiments/walk_forward_enhanced/:")
    print("  1. heatmap_all_horizons.png        - Heatmap of all returns")
    print("  2. line_chart_top_models.png       - Top 3 models per horizon")
    print("  3. boxplot_consistency.png         - Return distribution")
    print("  4. sequence_vs_traditional.png     - Sequence vs Traditional")
    print("  5. win_loss_pattern.png            - Win/Loss patterns")
    print("\n")

if __name__ == '__main__':
    main()
