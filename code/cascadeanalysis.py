import json
import os
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# --- Helper Functions for Metric Calculation ---

def get_cascade_metrics(node):
    """
    Calculates size, max_depth, and prepares data for the Wiener index calculation for a single cascade tree.
    This version is hardened against missing 'id' fields in the data.
    """
    # --- MODIFICATION START ---
    # Check if the root node itself is valid.
    root_id = node.get('id')
    if not root_id:
        print("Warning: Skipping a tree because its root node is missing an 'id'.")
        return 0, 0, [], {}
    # --- MODIFICATION END ---
        
    all_nodes = []
    adj_list = {}
    max_depth = 0
    
    q = deque([node])
    visited_ids = {root_id}
    
    while q:
        current_node = q.popleft()
        all_nodes.append(current_node)
        
        # --- MODIFICATION START ---
        node_id = current_node.get('id')
        if not node_id:
            # This case should be rare if the root check passes, but is good practice.
            print(f"Warning: Skipping a node within tree {root_id} because it is missing an 'id'.")
            continue
        # --- MODIFICATION END ---
        
        max_depth = max(max_depth, current_node.get('depth', 0))
        
        if node_id not in adj_list:
            adj_list[node_id] = []

        for child in current_node.get('children', []):
            # --- MODIFICATION START ---
            child_id = child.get('id')
            if not child_id:
                print(f"Warning: Skipping a child node in tree {root_id} because it is missing an 'id'.")
                continue
            # --- MODIFICATION END ---

            if child_id not in visited_ids:
                adj_list[node_id].append(child_id)
                if child_id not in adj_list:
                    adj_list[child_id] = []
                adj_list[child_id].append(node_id) # For an undirected graph
                
                visited_ids.add(child_id)
                q.append(child)
                
    return len(all_nodes), max_depth, all_nodes, adj_list


def calculate_wiener_index(adj_list):
    """
    Calculates the Wiener index (sum of shortest path distances between all node pairs).
    This is a measure of structural virality/compactness.
    """
    total_distance = 0
    nodes = list(adj_list.keys())
    
    if len(nodes) < 2:
        return 0

    # For each node, run a BFS to find distances to all other nodes
    for start_node in nodes:
        q = deque([(start_node, 0)])
        distances = {start_node: 0}
        
        while q:
            current_node, dist = q.popleft()
            
            for neighbor in adj_list[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    q.append((neighbor, dist + 1))
        
        total_distance += sum(distances.values())
        
    # The sum counts each pair twice (u to v and v to u), so we divide by 2.
    return total_distance / 2


def analyze_subreddit_cascades(input_filepath, results_dir, figures_dir):
    """
    Analyzes cascade shapes for a single subreddit.
    """
    subreddit_name = os.path.basename(input_filepath).replace('tree_', '').replace('.jsonl', '')
    print(f"--- Analyzing Cascades in {subreddit_name} ---")

    # --- Step 1: Load trees and compute metrics for each ---
    cascade_data = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # --- MODIFICATION START ---
                # Add a try-except block for each tree to handle malformed data gracefully
                # without stopping the analysis for the entire file.
                try:
                    tree = json.loads(line)
                    
                    is_toxic_seed = tree.get('toxicity_flag', False)
                    
                    size, max_depth, _, adj_list = get_cascade_metrics(tree)

                    if size == 0:
                        continue
                    
                    if size > 1:
                        wiener_index = calculate_wiener_index(adj_list)
                    else:
                        wiener_index = 0

                    cascade_data.append({
                        'thread_id': tree['id'], # This line can cause a KeyError
                        'is_toxic_seed': is_toxic_seed,
                        'size': size,
                        'max_depth': max_depth,
                        'wiener_index': wiener_index
                    })
                except KeyError as e:
                    print(f"Warning: A tree in {subreddit_name} is missing a required key: {e}. Skipping this tree.")
                    continue # Safely skip to the next tree
                # --- MODIFICATION END ---

    except FileNotFoundError:
        print(f"Error: Could not find file {input_filepath}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {input_filepath} on line {i+1}: {e}")
        return


    if not cascade_data:
        print(f"Warning: No valid cascades found for {subreddit_name}. Skipping analysis.")
        return

    df = pd.DataFrame(cascade_data)
    output_csv_path = os.path.join(results_dir, f'cascade_{subreddit_name}.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Saved computed metrics to {output_csv_path}")

    # --- Step 2: Perform Mann-Whitney U tests ---
    toxic_cascades = df[df['is_toxic_seed'] == True]
    nontoxic_cascades = df[df['is_toxic_seed'] == False]

    if toxic_cascades.empty or nontoxic_cascades.empty:
        print("Warning: Cannot perform comparison; only one type of cascade (toxic/non-toxic) found.")
        return
        
    test_results = {}
    metrics_to_test = ['size', 'max_depth', 'wiener_index']
    for metric in metrics_to_test:
        stat, p_value = mannwhitneyu(toxic_cascades[metric], nontoxic_cascades[metric], alternative='two-sided')
        test_results[metric] = {'U-statistic': stat, 'p-value': p_value}

    results_df = pd.DataFrame(test_results).T
    print("\nMann-Whitney U Test Results:")
    print(results_df)

    # Append results to the CSV file
    with open(output_csv_path, 'a') as f:
        f.write("\n\n--- Mann-Whitney U Test Results (Toxic vs. Non-Toxic Seeded) ---\n")
        results_df.to_csv(f)

    # --- Step 3: Create and save visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Cascade Shape Comparison in r/{subreddit_name}', fontsize=16)

    for i, metric in enumerate(metrics_to_test):
        sns.boxplot(ax=axes[i], x='is_toxic_seed', y=metric, data=df, showfliers=False) # Hiding outliers for readability
        axes[i].set_title(f'{metric.replace("_", " ").title()}\n(p-value: {test_results[metric]["p-value"]:.3f})')
        axes[i].set_xlabel("Is Root Post Toxic?")
        axes[i].set_ylabel("")

    axes[0].set_ylabel("Value")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(figures_dir, f'cascade_comparison_{subreddit_name}.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved comparison visualization to {fig_path}")


# --- Main execution block ---
if __name__ == '__main__':
    # Ensure you have the necessary libraries installed
    # pip install pandas scipy matplotlib seaborn
    
    network_data_dir = 'data/networks/'
    results_dir = 'results/'
    figures_dir = 'figures/'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    for filename in os.listdir(network_data_dir):
        if filename.startswith('tree_') and filename.endswith('.jsonl'):
            input_file = os.path.join(network_data_dir, filename)
            analyze_subreddit_cascades(input_file, results_dir, figures_dir)
            
    print("\nCascade analysis complete.")
