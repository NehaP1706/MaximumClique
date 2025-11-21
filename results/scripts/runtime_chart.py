#!/usr/bin/env python3
"""
generate_runtime_chart.py

This script processes the Maximum Clique experiment results and generates
a grouped bar chart comparing average runtime across algorithms and graph sizes.

Usage:
    python generate_runtime_chart.py [path_to_csv]

Input:
    - experiment_log.csv: CSV file containing experimental results

Output:
    - average_runtime_by_algorithm_and_size.csv: Processed data
    - runtime_comparison_chart.png: Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def find_csv_file():
    """
    Attempt to locate experiment_log.csv in common locations.

    Returns:
        Path to the CSV file
    """
    # Possible locations relative to the script
    possible_paths = [
        '../experiment_log.csv',  # Parent directory
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If not found, ask user
    return None

def load_and_clean_data(filename=None):
    """
    Load experiment data and remove invalid entries.

    Args:
        filename: Path to the experiment log CSV file (optional)

    Returns:
        DataFrame with clean, valid experimental results
    """
    if filename is None:
        filename = find_csv_file()

    if filename is None:
        print("\nError: Could not find experiment_log.csv")
        print("\nPlease provide the path to experiment_log.csv as an argument:")
        print("  python runtime_chart.py /path/to/experiment_log.csv")
        print("\nOr copy experiment_log.csv to one of these locations:")
        print("  - Same directory as this script")
        print("  - Parent directory (../)")
        print("  - ../../data/")
        sys.exit(1)

    print(f"Loading data from: {os.path.abspath(filename)}\n")

    # Read CSV with proper column names
    df = pd.read_csv(filename, header=None, 
                     names=['id', 'size_category', 'graph_file', 'num_nodes', 
                            'algorithm', 'clique_size', 'runtime'])

    # Remove rows with invalid data (marked as 'x' or 'xxxxxx')
    df_clean = df[~((df['clique_size'] == 'x') | (df['runtime'] == 'xxxxxx'))].copy()

    # Convert runtime to numeric
    df_clean['runtime'] = pd.to_numeric(df_clean['runtime'])

    print(f"Loaded {len(df)} total rows")
    print(f"Removed {len(df) - len(df_clean)} invalid rows")
    print(f"Processing {len(df_clean)} valid experimental results\n")

    return df_clean

def calculate_average_runtimes(df_clean):
    """
    Calculate average runtime for each algorithm-size combination.

    Args:
        df_clean: Clean DataFrame with experimental results

    Returns:
        DataFrame with average runtimes per algorithm and size category
    """
    # Calculate average runtime grouped by algorithm and size
    avg_runtime = df_clean.groupby(['algorithm', 'size_category'])['runtime'].mean().reset_index()

    # Pivot to create a more readable format
    chart_data = []
    for algorithm in avg_runtime['algorithm'].unique():
        algo_data = avg_runtime[avg_runtime['algorithm'] == algorithm]

        small = algo_data[algo_data['size_category'] == 'small']['runtime'].values
        medium = algo_data[algo_data['size_category'] == 'medium']['runtime'].values
        large = algo_data[algo_data['size_category'] == 'large']['runtime'].values

        chart_data.append({
            'algorithm': algorithm,
            'small': small[0] if len(small) > 0 else 0,
            'medium': medium[0] if len(medium) > 0 else 0,
            'large': large[0] if len(large) > 0 else 0
        })

    chart_df = pd.DataFrame(chart_data)

    # Save processed data
    output_csv = '../plots/average_runtime_by_algorithm_and_size.csv'
    chart_df.to_csv(output_csv, index=False)
    print("Average Runtime by Algorithm and Size Category:")
    print(chart_df.to_string(index=False))
    print(f"\nData saved to: {os.path.abspath(output_csv)}\n")

    return chart_df

def create_grouped_bar_chart(chart_df, output_filename='../plots/runtime_comparison_chart.png'):
    """
    Create and save a grouped bar chart visualization.

    Args:
        chart_df: DataFrame with average runtimes
        output_filename: Path for the output PNG file
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set width of bars and positions
    bar_width = 0.25
    algorithms = chart_df['algorithm']
    x = np.arange(len(algorithms))

    # Create bars for each size category
    bars1 = ax.bar(x - bar_width, chart_df['small'], bar_width, 
                   label='Small', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, chart_df['medium'], bar_width, 
                   label='Medium', color='#e67e22', alpha=0.8)
    bars3 = ax.bar(x + bar_width, chart_df['large'], bar_width, 
                   label='Large', color='#2ecc71', alpha=0.8)

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Labels and formatting
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Average Runtime Comparison Across Algorithms and Graph Sizes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(title='Graph Size', fontsize=10, title_fontsize=11)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {os.path.abspath(output_filename)}")

    # Don't show the chart (useful for scripts)
    # plt.show()
    plt.close()

def main():
    """Main execution function."""
    print("="*70)
    print("Maximum Clique Runtime Analysis - Chart Generator")
    print("="*70 + "\n")

    # Check if CSV path provided as argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Load and clean data
    df_clean = load_and_clean_data(csv_path)

    # Calculate averages
    chart_df = calculate_average_runtimes(df_clean)

    # Create visualization
    create_grouped_bar_chart(chart_df)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()