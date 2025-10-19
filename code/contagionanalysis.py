import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---

def flatten_tree_to_pairs(parent, pairs_list):
    """
    Recursively traverses a conversation tree and extracts parent-child pairs.
    This version is hardened to handle missing data keys.
    """
    # Use .get() to safely access parent's toxicity, defaulting to False.
    parent_toxic = parent.get('toxicity_flag', False)

    for child in parent.get('children', []):
        # Use .get() to safely access required child attributes.
        child_toxic = child.get('toxicity_flag', False)
        depth = child.get('depth')
        created_utc = child.get('created_utc')

        # Only append the pair if the essential data (depth, timestamp) is present.
        if depth is not None and created_utc is not None:
            pairs_list.append({
                'parent_toxic': int(parent_toxic), # Convert bool to 0/1
                'child_toxic': int(child_toxic),   # Convert bool to 0/1
                'depth': depth,
                'hour': datetime.fromtimestamp(created_utc).hour
            })
        
        # Continue traversal down the tree.
        flatten_tree_to_pairs(child, pairs_list)


def fit_and_summarize_model(df, formula):
    """Fits a logistic regression model and returns the summary and odds ratios."""
    try:
        model = sm.Logit.from_formula(formula, data=df).fit(disp=0)
        summary = model.summary2().tables[1]
        
        # Calculate Odds Ratios and Confidence Intervals
        params = model.params
        conf = model.conf_int()
        conf['Odds Ratio'] = params
        conf.columns = ['OR Conf. Int. Lower', 'OR Conf. Int. Upper', 'Odds Ratio']
        conf = conf.apply(np.exp)
        
        summary = summary.join(conf)
        return summary
    except Exception as e:
        print(f"  -> Could not fit model. Error: {e}")
        return pd.DataFrame() # Return empty dataframe on failure


def analyze_subreddit_contagion(input_filepath, results_dir, figures_dir):
    """
    Performs contagion analysis for a single subreddit.
    """
    subreddit_name = os.path.basename(input_filepath).replace('tree_', '').replace('.jsonl', '')
    print(f"--- Analyzing Contagion in {subreddit_name} ---")

    all_pairs = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Add a try-except block to handle malformed individual trees.
                try:
                    tree = json.loads(line)
                    flatten_tree_to_pairs(tree, all_pairs)
                except Exception as e:
                    print(f"Warning: Skipping a malformed tree in {subreddit_name} on line {i+1}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Could not find file {input_filepath}")
        return

    if not all_pairs:
        print(f"Warning: No valid parent-child pairs found for {subreddit_name}. Skipping.")
        return

    df = pd.DataFrame(all_pairs)
    
    # --- Data Validation ---
    if len(df['child_toxic'].unique()) < 2:
        print(f"Warning: Not enough variance in child toxicity for {subreddit_name}. Skipping regression analysis.")
        df.to_csv(os.path.join(results_dir, f'contagion_data_{subreddit_name}.csv'), index=False)
        print(f"Saved raw pair data to contagion_data_{subreddit_name}.csv")
        return

    # --- Step 1: Fit the main logistic regression model ---
    main_formula = 'child_toxic ~ parent_toxic + depth + C(hour)'
    print("Fitting main model...")
    main_results = fit_and_summarize_model(df, main_formula)

    # --- Step 2: Analyze decay by fitting models for each depth level ---
    print("Fitting models by depth level...")
    depth_results = {}
    formula_by_depth = 'child_toxic ~ parent_toxic + C(hour)'
    for depth_level in [1, 2, 3]:
        df_depth = df[df['depth'] == depth_level]
        if len(df_depth['child_toxic'].unique()) < 2 or df_depth.shape[0] < 20:
             print(f"  -> Skipping depth {depth_level} due to insufficient data.")
             continue
        depth_results[f'Depth {depth_level}'] = fit_and_summarize_model(df_depth, formula_by_depth)

    # --- Step 3: Save all results to a single CSV file ---
    output_csv_path = os.path.join(results_dir, f'contagion_{subreddit_name}.csv')
    with open(output_csv_path, 'w') as f:
        f.write("--- Main Model Results ---\n")
        if not main_results.empty:
            main_results.to_csv(f)
        
        for depth_level, results in depth_results.items():
            f.write(f"\n\n--- Results for {depth_level} ---\n")
            if not results.empty:
                results.to_csv(f)
    print(f"Saved all contagion analysis results to {output_csv_path}")

    # --- Step 4: Create and save visualizations ---
    plot_data = []
    if not main_results.empty and 'parent_toxic' in main_results.index:
        row = main_results.loc['parent_toxic']
        plot_data.append({
            'level': 'Overall',
            'OR': row['Odds Ratio'],
            'ci_lower': row['OR Conf. Int. Lower'],
            'ci_upper': row['OR Conf. Int. Upper']
        })

    for level, res in depth_results.items():
        if not res.empty and 'parent_toxic' in res.index:
            row = res.loc['parent_toxic']
            plot_data.append({
                'level': level,
                'OR': row['Odds Ratio'],
                'ci_lower': row['OR Conf. Int. Lower'],
                'ci_upper': row['OR Conf. Int. Upper']
            })
            
    if not plot_data:
        print("Could not generate plot as no valid model results for 'parent_toxic' were found.")
        return

    plot_df = pd.DataFrame(plot_data)
    plot_df['error'] = plot_df['ci_upper'] - plot_df['ci_lower']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(plot_df['level'], plot_df['OR'], 
                  yerr=plot_df['error']/2, # yerr is half the total error range
                  capsize=5, color='skyblue', alpha=0.8)

    ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='No Effect (OR = 1)')
    
    ax.set_ylabel('Odds Ratio of Child Toxicity', fontsize=12)
    ax.set_xlabel('Model Scope', fontsize=12)
    ax.set_title(f'Effect of Parent\'s Toxicity on Child\'s Toxicity in r/{subreddit_name}', fontsize=14)
    ax.legend()
    
    # Add labels to bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f'contagion_odds_ratios_{subreddit_name}.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved contagion visualization to {fig_path}")


# --- Main execution block ---
if __name__ == '__main__':
    # Ensure you have the necessary libraries installed
    # pip install pandas statsmodels matplotlib seaborn numpy
    
    network_data_dir = 'data/networks/'
    results_dir = 'results/'
    figures_dir = 'figures/'
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    for filename in os.listdir(network_data_dir):
        if filename.startswith('tree_') and filename.endswith('.jsonl'):
            input_file = os.path.join(network_data_dir, filename)
            analyze_subreddit_contagion(input_file, results_dir, figures_dir)
            
    print("\nContagion analysis complete.")