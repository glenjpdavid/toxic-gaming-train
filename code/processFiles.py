# Hi! Here is the code we used to combine comments.jsonl and posts.jsonl together!


import json, re, csv, sys, os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# Here we set the file name the code needs to process
COMMENTS_JSONL = "r_Ninesols_comments.jsonl"
POSTS_JSONL    = "r_Ninesols_posts.jsonl"
OUTPUT_CSV     = "r_Ninesols_comments.csv"

REQUIRED_COLS = [
    "subreddit", "post_id", "title", "comment_id", "parent_id",
    "author", "body", "created_utc", "score", "flair", "link_id"
]

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def derive_post_id_from_post(obj: Dict[str, Any]) -> Optional[str]:
    # Prefer 'id'
    if isinstance(obj.get("id"), str) and obj.get("id"):
        return obj["id"]
    # Fallback to 'name' like t3_xxxxx
    name = obj.get("name")
    if isinstance(name, str) and name.startswith("t3_"):
        return name.split("_", 1)[1]
    # Parse from permalink or url
    for k in ("permalink", "url", "url_overridden_by_dest", "full_link"):
        v = obj.get(k)
        if isinstance(v, str):
            m = re.search(r"/comments/([a-z0-9]+)/", v)
            if m:
                return m.group(1)
    return None

def derive_post_id_from_comment(obj: Dict[str, Any]) -> Optional[str]:
    link_id = obj.get("link_id")
    if isinstance(link_id, str) and link_id.startswith("t3_"):
        return link_id.split("_", 1)[1]
    return None

def coerce_flair(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("flair", "author_flair_text", "link_flair_text", "flair_text"):
        v = obj.get(k)
        if v not in (None, "", "null"):
            return v
    return None

def load_posts(posts_path: str) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for p in read_jsonl(posts_path):
        recs.append({
            "post_id": derive_post_id_from_post(p),
            "title": p.get("title"),
            "subreddit_post": p.get("subreddit"),
        })
    return pd.DataFrame(recs)

def load_comments(comments_path: str) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for c in read_jsonl(comments_path):
        recs.append({
            "subreddit": c.get("subreddit"),
            "post_id": derive_post_id_from_comment(c),
            "comment_id": c.get("id") or c.get("comment_id"),
            "parent_id": c.get("parent_id"),
            "author": c.get("author"),
            "body": c.get("body") or c.get("selftext"),
            "created_utc": c.get("created_utc") or c.get("created"),
            "score": c.get("score"),
            "flair": coerce_flair(c),
            "link_id": c.get("link_id")
        })
    return pd.DataFrame(recs)

def process(posts_path: str, comments_path: str, out_csv: str):
    df_posts = load_posts(posts_path)
    df_comments = load_comments(comments_path)

    # Join
    merged = pd.merge(
        df_comments,
        df_posts[["post_id", "title"]],
        on="post_id",
        how="left"
    )

    # Reorder and ensure columns exist
    for col in REQUIRED_COLS:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[REQUIRED_COLS]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Write CSV
    merged.to_csv(out_csv, index=False, encoding="utf-8")

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
    comments_path, posts_path, out_csv = resolve_paths_from_args_or_config(sys.argv)
    # Basic validation
    if not os.path.isfile(comments_path):
        print(f"Comments file not found: {comments_path}")
        sys.exit(1)
    if not os.path.isfile(posts_path):
        print(f"Posts file not found: {posts_path}")
        sys.exit(1)
    process(posts_path, comments_path, out_csv)
    print(f"Wrote CSV: {out_csv}")
