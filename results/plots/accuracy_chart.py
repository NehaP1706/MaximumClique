#!/usr/bin/env python3
"""
generate_accuracy_chart.py

Process experiment results and generate a grouped bar chart comparing
average accuracy of each algorithm relative to `bron_kerbosch` (treated
as the baseline optimal solution when available).

Usage:
    python accuracy_chart.py [path_to_csv]

Outputs:
    - average_accuracy_by_algorithm_and_size.csv
    - accuracy_comparison_chart.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def find_csv_file():
    possible_paths = [
        'experiment_log.csv',
        'results/experiment_log.csv',
        '../experiment_log.csv',
        '../../experiment_log.csv',
        '../data/experiment_log.csv',
        '../../data/experiment_log.csv',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_and_clean_data(filename=None):
    if filename is None:
        filename = find_csv_file()

    if filename is None:
        print("Error: Could not find experiment_log.csv")
        print("Provide path as argument: python accuracy_chart.py /path/to/experiment_log.csv")
        sys.exit(1)

    print(f"Loading data from: {os.path.abspath(filename)}")

    df = pd.read_csv(filename, header=None,
                     names=['id', 'size_category', 'graph_file', 'num_nodes',
                            'algorithm', 'clique_size', 'runtime'])

    # Remove rows where clique size is missing ('x')
    df_clean = df[df['clique_size'] != 'x'].copy()

    # Convert clique_size to numeric
    df_clean['clique_size'] = pd.to_numeric(df_clean['clique_size'])

    print(f"Loaded {len(df)} rows; {len(df_clean)} rows with valid clique sizes")
    return df_clean


def compute_accuracy(df_clean):
    # Map each experiment id to bron_kerbosch clique size (baseline)
    bron = df_clean[df_clean['algorithm'] == 'bron_kerbosch'][['id', 'clique_size']].copy()
    bron = bron.rename(columns={'clique_size': 'bron_clique_size'})

    # Merge baseline into full dataframe on 'id'
    merged = pd.merge(df_clean, bron, on='id', how='left')

    # Only compute accuracy where bron_kerbosch baseline is present
    merged = merged[~merged['bron_clique_size'].isna()].copy()

    # Compute accuracy as ratio; clip at 1.0
    merged['accuracy'] = (merged['clique_size'] / merged['bron_clique_size']).clip(upper=1.0)

    # Exclude bron_kerbosch rows from algorithm comparisons (would be 1.0)
    # but keep them for completeness if desired; we'll include them but they will be 1.0

    return merged


def calculate_average_accuracies(merged):
    # Group by algorithm and size_category and compute mean accuracy
    avg_accuracy = merged.groupby(['algorithm', 'size_category'])['accuracy'].mean().reset_index()

    chart_data = []
    for algorithm in avg_accuracy['algorithm'].unique():
        algo_data = avg_accuracy[avg_accuracy['algorithm'] == algorithm]

        small = algo_data[algo_data['size_category'] == 'small']['accuracy'].values
        medium = algo_data[algo_data['size_category'] == 'medium']['accuracy'].values
        large = algo_data[algo_data['size_category'] == 'large']['accuracy'].values

        chart_data.append({
            'algorithm': algorithm,
            'small': float(small[0]) if len(small) > 0 else 0.0,
            'medium': float(medium[0]) if len(medium) > 0 else 0.0,
            'large': float(large[0]) if len(large) > 0 else 0.0,
        })

    chart_df = pd.DataFrame(chart_data)

    output_csv = os.path.join(os.path.dirname(__file__), 'average_accuracy_by_algorithm_and_size.csv')
    chart_df.to_csv(output_csv, index=False)
    print('\nAverage Accuracy by Algorithm and Size Category:')
    print(chart_df.to_string(index=False))
    print(f'\nData saved to: {os.path.abspath(output_csv)}\n')

    return chart_df


def create_grouped_bar_chart(chart_df, output_filename=None):
    if output_filename is None:
        output_filename = os.path.join(os.path.dirname(__file__), 'accuracy_comparison_chart.png')
    
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.25
    algorithms = chart_df['algorithm']
    x = np.arange(len(algorithms))

    bars1 = ax.bar(x - bar_width, chart_df['small'], bar_width,
                   label='Small', color='#3498db', alpha=0.9)
    bars2 = ax.bar(x, chart_df['medium'], bar_width,
                   label='Medium', color='#e67e22', alpha=0.9)
    bars3 = ax.bar(x + bar_width, chart_df['large'], bar_width,
                   label='Large', color='#2ecc71', alpha=0.9)

    ax.set_ylim(0, 1.02)

    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (fraction of optimal)', fontsize=12, fontweight='bold')
    ax.set_title('Average Accuracy Compared to Bron-Kerbosch (per Graph Size)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(title='Graph Size', fontsize=10, title_fontsize=11)

    # Annotate bars with percentage labels
    def annotate_bars(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f'Chart saved to: {os.path.abspath(output_filename)}')
    plt.close()


def main():
    print('='*70)
    print('Maximum Clique Accuracy Analysis - Chart Generator')
    print('='*70 + '\n')

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None

    df_clean = load_and_clean_data(csv_path)
    merged = compute_accuracy(df_clean)

    print(f"Computing accuracies using {len(merged)} rows with bron_kerbosch baseline present")

    chart_df = calculate_average_accuracies(merged)
    create_grouped_bar_chart(chart_df)

    print('\n' + '='*70)
    print('Analysis complete!')
    print('='*70)


if __name__ == '__main__':
    main()
