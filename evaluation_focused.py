import os
import subprocess
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def run_single_backtest(symbol, venv_python, iteration, params, experiment_num):
    """
    Runs a single instance of backtesting.py with a given set of parameters.
    """
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    print(f"--- Running Exp {experiment_num}, Iteration {iteration+1} ({param_str}) ---")
    
    output_dir = 'evaluation_focused_results' # New directory for results
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'exp{experiment_num}_iter{iteration}.json')

    cmd = [venv_python, "backtesting.py", symbol, "--json_output", json_path]
    
    for key, value in params.items():
        cmd.append(f"--{key}")
        if isinstance(value, list):
            for v in value:
                cmd.append(str(v))
        else:
            cmd.append(str(value))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Iteration {iteration+1} complete.")
        return json_path, params
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Backtesting script failed for iteration {iteration+1}.")
        print(f"Stderr: {e.stderr}")
        return None, None

def analyze_results(results_data):
    """
    Analyzes the collected results and generates a summary and plot.
    """
    if not results_data:
        print("No results to analyze.")
        return

    records = []
    for result in results_data:
        with open(result['file'], 'r') as f:
            data = json.load(f)
            param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
            for strategy, equity in data.items():
                records.append({'params': param_str, 'strategy': strategy, 'final_equity': equity})
    
    df = pd.DataFrame(records)

    print("\n\n--- Focused Evaluation Summary ---")
    summary = df.groupby(['params', 'strategy'])['final_equity'].agg(
        ['mean', 'std', lambda x: (x > 1.0).mean() * 100]
    ).rename(columns={'mean': 'Avg Equity', 'std': 'Std Dev', '<lambda_0>': 'Win Rate (%)'})
    
    summary['Sharpe Ratio'] = (summary['Avg Equity'] - 1) / summary['Std Dev']
    
    print(summary.to_string(float_format="%.4f"))

    main_strategies = [s for s in df['strategy'].unique() if 'N-BEATS' in s]
    if not main_strategies:
        print("\nNo 'N-BEATS' strategies found to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    sns.boxplot(data=df[df['strategy'].isin(main_strategies)], x='params', y='final_equity', hue='strategy', ax=ax)
    
    ax.axhline(1.0, color='red', linestyle='--', label='Break-even (1.0)')
    ax.set_title(f'Focused Evaluation: Distribution of Final Equity', fontsize=16)
    ax.set_xlabel('Experiment Parameters', fontsize=12)
    ax.set_ylabel('Final Equity', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Strategy')
    plt.tight_layout()
    
    plot_filename = 'evaluation_focused_summary.png' # New plot filename
    plt.savefig(plot_filename)
    print(f"\nSaved summary box plot to {plot_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a focused evaluation on promising backtest parameters.")
    parser.add_argument("symbol", type=str, help="The currency symbol to test (e.g., XNO_USDT).")
    parser.add_argument("num_iterations", type=int, help="The number of random backtests to run for EACH parameter set.")
    args = parser.parse_args()

    # --- FOCUSED PARAMETER GRID ---
    # Here we test variations around the winning parameters (ws=60, f=1).
    param_grid = [
        # The winning baseline from the last run
        {'window_sizes': [60], 'forecast_intervals': [1], 'epochs': 15, 'test_size': 0.2, 'timehorizon': 1000},
        
        # Test 1: Increase window size further. Does more context help?
        {'window_sizes': [90], 'forecast_intervals': [1], 'epochs': 15, 'test_size': 0.2, 'timehorizon': 1000},

        # Test 2: Use more data for training. Does a larger dataset improve robustness?
        {'window_sizes': [60], 'forecast_intervals': [1], 'epochs': 15, 'test_size': 0.2, 'timehorizon': 2000},

        # Test 3: Change the train/test split. Is performance sensitive to this?
        {'window_sizes': [60], 'forecast_intervals': [1], 'epochs': 15, 'test_size': 0.3, 'timehorizon': 1000},
    ]

    import sys
    venv_python = sys.executable

    all_results_data = []
    for i, params in enumerate(param_grid):
        for j in range(args.num_iterations):
            result_file, used_params = run_single_backtest(args.symbol, venv_python, j, params, i)
            if result_file:
                all_results_data.append({'file': result_file, 'params': used_params})

    if all_results_data:
        analyze_results(all_results_data)
    else:
        print("\nNo backtests completed successfully.")