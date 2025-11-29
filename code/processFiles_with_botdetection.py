import json
import re
import csv
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# --- Configuration ---
COMMENTS_JSONL = "r_Ninesols_comments.jsonl"
POSTS_JSONL    = "r_Ninesols_posts.jsonl"
OUTPUT_DIR     = "cleaned_data" # New folder for cleaned output
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "r_Ninesols_cleaned.csv") # Updated path

# Columns required in the final CSV, now including the 'is_bot' tag.
REQUIRED_COLS = [
    "subreddit", "post_id", "title", "comment_id", "parent_id",
    "author", "body", "created_utc", "score", "flair", "link_id",
    "is_bot"  # New column for tagging content
]

# --- Bot Detection Heuristics ---
# 1. Known authors (e.g., subreddit moderators who post automatically)
KNOWN_BOT_AUTHORS = {
    "[deleted]", 
    "AutoModerator", 
    "TrollaBot", 
    "RemindMeBot",
    "BotDefense",
    # Add other known bot usernames here if needed
}

# 2. NLP/Keyword patterns (case-insensitive for body/title checks)
BOT_KEYWORDS = [
    "i am a bot", 
    "this action was performed automatically", 
    "beep boop", 
    "good bot", 
    "bad bot",
    "source code is here",
    "was summoned by",
]
# Create a single regex pattern for efficiency
BOT_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, BOT_KEYWORDS)) + r')\b', re.IGNORECASE)


def read_jsonl(path: str):
    """Reads a JSONL file line by line and yields parsed objects."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        sys.exit(1)


# --- Helper Functions (Unchanged from original code) ---

def derive_post_id_from_post(obj: Dict[str, Any]) -> Optional[str]:
    """Derives a standard post ID (e.g., 'xxxxx') from various post fields."""
    if isinstance(obj.get("id"), str) and obj.get("id"):
        return obj["id"]
    name = obj.get("name")
    if isinstance(name, str) and name.startswith("t3_"):
        return name.split("_", 1)[1]
    for k in ("permalink", "url", "url_overridden_by_dest", "full_link"):
        v = obj.get(k)
        if isinstance(v, str):
            m = re.search(r"/comments/([a-z0-9]+)/", v)
            if m:
                return m.group(1)
    return None

def derive_post_id_from_comment(obj: Dict[str, Any]) -> Optional[str]:
    """Derives the post ID from a comment's link_id field (e.g., 't3_xxxxx')."""
    link_id = obj.get("link_id")
    if isinstance(link_id, str) and link_id.startswith("t3_"):
        return link_id.split("_", 1)[1]
    return None

def coerce_flair(obj: Dict[str, Any]) -> Optional[str]:
    """Attempts to find any relevant flair text in the object."""
    for k in ("flair", "author_flair_text", "link_flair_text", "flair_text"):
        v = obj.get(k)
        if v not in (None, "", "null"):
            return v
    return None

# --- Bot Filtering Logic ---

def check_for_bot_author(author: Optional[str]) -> bool:
    """Checks if the author is in the list of known bot usernames."""
    if not author:
        return False
    return author.lower() in {a.lower() for a in KNOWN_BOT_AUTHORS}

def check_for_bot_keywords(text: Optional[str]) -> bool:
    """Uses regex to check if the text contains common bot-related phrases."""
    if not text or text in ("[removed]", "[deleted]"):
        return False
    # Check for exact matches to the predefined patterns
    return bool(BOT_PATTERN.search(text))

def tag_as_bot(row: Dict[str, Any]) -> bool:
    """Applies all bot heuristics to a single record (post or comment)."""
    # Heuristic 1: Known Author
    if check_for_bot_author(row.get("author")):
        return True
    
    # Heuristic 2: Textual Identification
    content = row.get("body") or row.get("title") or ""
    if check_for_bot_keywords(content):
        return True

    # Heuristic 3: Flair-based (often bots have 'BOT' or 'AUTOMODERATOR' in flair)
    flair = row.get("flair")
    if isinstance(flair, str) and ("bot" in flair.lower() or "automod" in flair.lower()):
        return True

    return False

# --- Data Loading Functions (Updated to include body/selftext and tagging) ---

def load_posts(posts_path: str) -> pd.DataFrame:
    """Loads posts, extracts relevant fields, and applies the bot tag."""
    recs: List[Dict[str, Any]] = []
    for p in read_jsonl(posts_path):
        record = {
            "post_id": derive_post_id_from_post(p),
            "title": p.get("title"),
            "author": p.get("author"),
            # Use selftext for the main content of a post
            "body": p.get("selftext") or p.get("body") or "", 
            "subreddit_post": p.get("subreddit"),
            # Placeholder for comment-specific fields in the post DF
            "comment_id": None,
            "parent_id": None,
            "link_id": None,
            "created_utc": p.get("created_utc") or p.get("created"),
            "score": p.get("score"),
            "flair": coerce_flair(p),
            "subreddit": p.get("subreddit"), # Store subreddit here for consistency
        }
        record["is_bot"] = tag_as_bot(record) # Tag the record
        recs.append(record)
        
    df = pd.DataFrame(recs)
    # Ensure all required columns are present for merging later
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None
            
    # Keep only columns necessary for merging and final output (posts can be bot-filtered too)
    return df[REQUIRED_COLS]

def load_comments(comments_path: str) -> pd.DataFrame:
    """Loads comments, extracts relevant fields, and applies the bot tag."""
    recs: List[Dict[str, Any]] = []
    for c in read_jsonl(comments_path):
        record = {
            "subreddit": c.get("subreddit"),
            "post_id": derive_post_id_from_comment(c),
            "comment_id": c.get("id") or c.get("comment_id"),
            "parent_id": c.get("parent_id"),
            "author": c.get("author"),
            # Use body for comment content
            "body": c.get("body") or c.get("selftext") or "", 
            "created_utc": c.get("created_utc") or c.get("created"),
            "score": c.get("score"),
            "flair": coerce_flair(c),
            "link_id": c.get("link_id")
        }
        record["is_bot"] = tag_as_bot(record) # Tag the record
        recs.append(record)
        
    df = pd.DataFrame(recs)
    # Ensure all required columns are present
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None
            
    return df

# --- Main Processing Logic (Updated for filtering) ---

def process(posts_path: str, comments_path: str, out_csv: str):
    """
    Loads posts and comments, tags bot-generated content,
    removes bot content, merges the datasets, and exports to CSV.
    """
    print("Loading and tagging posts...")
    df_posts_full = load_posts(posts_path)
    print(f"Total posts loaded: {len(df_posts_full)}")
    
    print("Loading and tagging comments...")
    df_comments_full = load_comments(comments_path)
    print(f"Total comments loaded: {len(df_comments_full)}")

    # 1. Filter out bot content (as requested)
    df_posts_human = df_posts_full[df_posts_full['is_bot'] == False].copy()
    df_comments_human = df_comments_full[df_comments_full['is_bot'] == False].copy()

    print(f"Human posts remaining: {len(df_posts_human)}")
    print(f"Human comments remaining: {len(df_comments_human)}")
    
    # 2. Extract necessary post info for merging (Post ID and Title)
    post_info = df_posts_human[["post_id", "title"]].drop_duplicates(subset=["post_id"])

    # 3. Join human comments with human post titles
    # We will join df_comments_human with the titles from df_posts_human
    print("Merging comments with post titles...")
    merged_comments = pd.merge(
        df_comments_human,
        post_info,
        on="post_id",
        how="left",
        suffixes=('_comment', '_post_info')
    )
    
    # 4. Combine all human data (comments + posts)
    # Create a unified DataFrame by appending the human posts 
    final_df = pd.concat([merged_comments, df_posts_human], ignore_index=True)

    # 5. Reorder and ensure columns exist (using the full set of REQUIRED_COLS)
    for col in REQUIRED_COLS:
        if col not in final_df.columns:
            final_df[col] = None
    
    # Ensure the final columns are in the correct order
    final_df = final_df[REQUIRED_COLS]
    
    # Final cleanup: drop rows where both post_id and comment_id are missing (should be rare)
    final_df = final_df.dropna(subset=['post_id', 'comment_id'], how='all')

    # Ensure output directory exists. This now uses the path from OUTPUT_CSV 
    # (e.g., "cleaned_data/r_Ninesols_cleaned.csv") and creates the "cleaned_data" folder.
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # 6. Write CSV
    final_df.to_csv(out_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

def resolve_paths_from_args_or_config(argv: List[str]) -> Tuple[str, str, str]:
    """
    If CLI args are provided, use them.
    Otherwise fall back to CONFIG paths above.
    """
    if len(argv) >= 4:
        comments, posts, out_csv = argv[1], argv[2], argv[3]
        return comments, posts, out_csv
    # No CLI, use config
    return COMMENTS_JSONL, POSTS_JSONL, OUTPUT_CSV

if __name__ == "__main__":
    # Check if we are running in a simulated environment without files
    if not os.path.exists(COMMENTS_JSONL) and not os.path.exists(POSTS_JSONL):
         print("Warning: Input files not found. Using default names but assuming data files will be present at runtime.")

    comments_path, posts_path, out_csv = resolve_paths_from_args_or_config(sys.argv)
    
    # In a real environment, basic file validation would be here:
    # if not os.path.isfile(comments_path):
    #     print(f"Comments file not found: {comments_path}")
    #     sys.exit(1)
    # if not os.path.isfile(posts_path):
    #     print(f"Posts file not found: {posts_path}")
    #     sys.exit(1)
        
    process(posts_path, comments_path, out_csv)
    print(f"Wrote CSV with human-generated content to: {out_csv}")