import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from tqdm import tqdm

# Configuration
INPUT_PATH = "*_botfiltered.csv"
RESULTS_DIR = "results_research_analysis"
FIGURES_DIR = "figures_research_analysis"
CASCADE_DIR = "results_cascade_analysis"  # Directory containing cascade metrics
MIN_INTERACTIONS = 5

# Classification Mapping
GAME_CLASSIFICATION = {
    'wow': 'AAA MMORPG',
    'elderscrollsonline': 'AAA MMORPG',
    'darksouls': 'AAA Soulslike',
    'liesofp': 'AAA Soulslike', # Included as requested
    'albiononline': 'Indie MMORPG',
    'palia': 'Indie MMORPG',
    'hollowknight': 'Indie Soulslike',
    'ninesols': 'Indie Soulslike', # Inferred from upload
    'r_gaming': 'Baseline'
}

def setup_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

def clean_id(id_val):
    if pd.isna(id_val): return id_val
    return str(id_val).split('_')[-1]

def load_and_prep_data(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # ID Standardization
        if 'comment_id' in df.columns and 'post_id' in df.columns:
            df['id'] = df['comment_id'].fillna(df['post_id'])
        elif 'comment_id' in df.columns:
            df['id'] = df['comment_id']
        elif 'post_id' in df.columns:
            df['id'] = df['post_id']
        elif 'id' not in df.columns:
            return None
            
        df['id'] = df['id'].apply(clean_id)
        df['parent_id'] = df['parent_id'].apply(clean_id)
        
        # Toxicity Standardization
        if 'moderation_flagged' in df.columns:
            df['is_toxic'] = df['moderation_flagged']
        elif 'openai_flag' in df.columns:
            df['is_toxic'] = df['openai_flag']
        elif 'score' in df.columns and df['score'].max() <= 1.0:
            df['is_toxic'] = (df['score'] > 0.5)
        else:
            df['is_toxic'] = 0

        df['is_toxic'] = df['is_toxic'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype(int)
        
        # Hour
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
            df['hour'] = df['datetime'].dt.hour
        else:
            df['hour'] = 0
            
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_subreddit(filepath):
    subreddit = os.path.basename(filepath).replace('_botfiltered.csv', '').lower()
    
    # Map to Category
    category = GAME_CLASSIFICATION.get(subreddit, 'Unclassified')
    if category == 'Unclassified':
        # Fuzzy match attempt
        for key in GAME_CLASSIFICATION:
            if key in subreddit:
                category = GAME_CLASSIFICATION[key]
                break
    
    print(f"Processing {subreddit} ({category})...")
    df = load_and_prep_data(filepath)
    if df is None or df.empty: return None

    # --- RQ3: Base Toxicity Rate ---
    total_comments = len(df)
    toxic_comments = df['is_toxic'].sum()
    toxicity_rate = (toxic_comments / total_comments) * 100

    # --- RQ1: Content Contagion (Odds Ratio) ---
    # Build Dyads
    parents = df[['id', 'is_toxic']].copy()
    parents.columns = ['parent_id', 'parent_is_toxic']
    dyads = df.merge(parents, on='parent_id', how='inner')
    
    odds_ratio = np.nan
    p_value_or = np.nan
    
    if len(dyads) > 50:
        try:
            # We add C(hour) to control for time
            model = smf.logit("is_toxic ~ parent_is_toxic + C(hour)", data=dyads).fit(disp=0)
            odds_ratio = np.exp(model.params['parent_is_toxic'])
            p_value_or = model.pvalues['parent_is_toxic']
        except:
            pass

    # --- RQ1: User Contagion (Delta) ---
    # We need Author info
    user_delta = np.nan
    
    if 'author' in dyads.columns:
        # Prepare User Dyads (need parent author too)
        # Drop temp columns to avoid merge conflict
        cols_to_drop = ['parent_author']
        clean_dyads = dyads.drop(columns=[c for c in cols_to_drop if c in dyads.columns])
        
        parent_authors = df[['id', 'author']].copy()
        parent_authors.columns = ['parent_id', 'parent_author']
        
        user_dyads = clean_dyads.merge(parent_authors, on='parent_id', how='inner')
        user_dyads = user_dyads[~user_dyads['author'].isin(['[deleted]', '[removed]', np.nan])]
        
        # Filter active users
        author_counts = user_dyads['author'].value_counts()
        active = author_counts[author_counts >= MIN_INTERACTIONS].index
        active_dyads = user_dyads[user_dyads['author'].isin(active)]
        
        if not active_dyads.empty:
            user_stats = active_dyads.groupby(['author', 'parent_is_toxic'])['is_toxic'].mean().unstack()
            # Must have replied to both (0 and 1)
            if 0 in user_stats.columns and 1 in user_stats.columns:
                valid_users = user_stats.dropna()
                if not valid_users.empty:
                    deltas = valid_users[1] - valid_users[0]
                    user_delta = deltas.mean()
    
    # --- RQ1 & RQ3: Cascade Analysis Integration ---
    # Load pre-computed cascade metrics
    avg_toxic_depth = np.nan
    avg_toxic_size = np.nan
    avg_toxic_virality = np.nan
    
    # Try to find the matching cascade file (case insensitive match)
    # File pattern: results_cascade_analysis/{subreddit}_cascade_metrics.csv
    cascade_file_pattern = os.path.join(CASCADE_DIR, f"{subreddit}_cascade_metrics.csv")
    
    # Handle potential casing differences in filename (e.g. AlbionOnline vs albiononline)
    cascade_matches = glob.glob(os.path.join(CASCADE_DIR, "*.csv"))
    found_cascade_file = None
    for cf in cascade_matches:
        base = os.path.basename(cf).lower()
        if f"{subreddit}_cascade_metrics.csv" == base:
            found_cascade_file = cf
            break
            
    if found_cascade_file and os.path.exists(found_cascade_file):
        try:
            cascades = pd.read_csv(found_cascade_file)
            # Filter for TOXIC ROOTS only (as per RQ1: "longer toxic threads")
            toxic_cascades = cascades[cascades['is_root_toxic'] == 1]
            
            if not toxic_cascades.empty:
                avg_toxic_depth = toxic_cascades['max_depth'].mean()
                avg_toxic_size = toxic_cascades['size'].mean()
                # Virality can be 0 or NaN for small trees, filter NaNs
                avg_toxic_virality = toxic_cascades['structural_virality'].dropna().mean()
        except Exception as e:
            print(f"  Could not load cascade metrics for {subreddit}: {e}")
    else:
        print(f"  No cascade metrics file found for {subreddit}")

    return {
        'Subreddit': subreddit,
        'Category': category,
        'Toxicity_Rate': toxicity_rate,
        'Contagion_OR': odds_ratio,
        'Contagion_P_Value': p_value_or,
        'User_Delta': user_delta,
        'Avg_Toxic_Depth': avg_toxic_depth,
        'Avg_Toxic_Size': avg_toxic_size,
        'Avg_Toxic_Virality': avg_toxic_virality
    }

def generate_comparative_plots(results_df):
    sns.set_style("whitegrid")
    
    # Order for plots
    order = ['Baseline', 'AAA MMORPG', 'Indie MMORPG', 'AAA Soulslike', 'Indie Soulslike']
    # Filter order to only include categories that exist in results
    existing_order = [c for c in order if c in results_df['Category'].unique()]
    
    # 1. RQ3: Toxicity Rates
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Category', y='Toxicity_Rate', order=existing_order, palette='viridis', errorbar=None)
    plt.title("RQ3: Average Toxicity Rate by Genre & Scale")
    plt.ylabel("Toxicity Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/RQ3_Toxicity_Rates.png")
    plt.close()

    # 2. RQ1: Contagion Odds Ratio
    plt.figure(figsize=(10, 6))
    # Filter out NaNs
    or_df = results_df.dropna(subset=['Contagion_OR'])
    sns.barplot(data=or_df, x='Category', y='Contagion_OR', order=existing_order, palette='magma', errorbar=None)
    plt.axhline(1, color='red', linestyle='--', label='No Contagion')
    plt.title("RQ1: Contagion Odds Ratio (Likelihood of Toxic Reply to Toxic Parent)")
    plt.ylabel("Odds Ratio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/RQ1_Contagion_OR.png")
    plt.close()

    # 3. RQ1: User Delta
    plt.figure(figsize=(10, 6))
    delta_df = results_df.dropna(subset=['User_Delta'])
    if not delta_df.empty:
        sns.barplot(data=delta_df, x='Category', y='User_Delta', order=existing_order, palette='coolwarm', errorbar=None)
        plt.axhline(0, color='black', linestyle='--', label='No User Effect')
        plt.title("RQ1: User Contagion Delta (Propensity to Retaliate)")
        plt.ylabel("Delta (P(T|T) - P(T|NT))")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/RQ1_User_Delta.png")
        plt.close()

    # 4. RQ1: Average Toxic Thread Depth
    plt.figure(figsize=(10, 6))
    depth_df = results_df.dropna(subset=['Avg_Toxic_Depth'])
    if not depth_df.empty:
        sns.barplot(data=depth_df, x='Category', y='Avg_Toxic_Depth', order=existing_order, palette='Reds', errorbar=None)
        plt.title("RQ1: Avg Depth of Toxic Threads (Persistence)")
        plt.ylabel("Max Depth")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/RQ1_Cascade_Depth.png")
        plt.close()

    # 5. RQ1: Average Toxic Thread Size
    plt.figure(figsize=(10, 6))
    size_df = results_df.dropna(subset=['Avg_Toxic_Size'])
    if not size_df.empty:
        sns.barplot(data=size_df, x='Category', y='Avg_Toxic_Size', order=existing_order, palette='Oranges', errorbar=None)
        plt.title("RQ1: Avg Size of Toxic Threads (Volume)")
        plt.ylabel("Number of Comments")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/RQ1_Cascade_Size.png")
        plt.close()

    # 6. RQ1: Average Toxic Thread Virality
    plt.figure(figsize=(10, 6))
    virality_df = results_df.dropna(subset=['Avg_Toxic_Virality'])
    if not virality_df.empty:
        sns.barplot(data=virality_df, x='Category', y='Avg_Toxic_Virality', order=existing_order, palette='Purples', errorbar=None)
        plt.title("RQ1: Avg Structural Virality of Toxic Threads")
        plt.ylabel("Wiener Index (Avg Distance)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/RQ1_Cascade_Virality.png")
        plt.close()

def main():
    setup_directories()
    files = glob.glob(INPUT_PATH)
    results = []
    
    for f in files:
        res = analyze_subreddit(f)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{RESULTS_DIR}/comparative_summary.csv", index=False)
        print("\n--- Comparative Summary ---")
        # Print columns including new cascade metrics
        cols = ['Subreddit', 'Category', 'Toxicity_Rate', 'Contagion_OR', 'User_Delta', 'Avg_Toxic_Depth', 'Avg_Toxic_Size']
        # Handle case where some cols might be all NaN if cascade file missing
        print(df[[c for c in cols if c in df.columns]].to_string())
        
        generate_comparative_plots(df)
        print(f"\nAnalysis complete. Figures saved to {FIGURES_DIR}/")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()