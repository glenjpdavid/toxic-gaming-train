import json
import os
from collections import defaultdict

def build_conversation_trees(input_filepath, output_filepath):
    """
    Reads a JSONL file of Reddit posts/comments, reconstructs conversation trees,
    and writes each tree as a new line in an output JSONL file.

    Each node in the tree will have its original data plus depth and timestamp.
    """
    nodes = {}
    
    # --- Step 1: Read all posts and comments into a dictionary for quick access ---
    # We use the unique 'id' of a post or comment as the key.
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    
                    # --- MODIFIED SECTION START ---
                    # Adapt to the user's data structure which uses 'post_id' and 'comment_id'.
                    record_type = item.get('record_type')
                    unique_id = None
                    if record_type == 'comment':
                        unique_id = item.get('comment_id')
                    elif record_type == 'post':
                        unique_id = item.get('post_id')

                    if not unique_id:
                        print(f"Warning: Could not find 'comment_id' or 'post_id' on line {line_num} in {input_filepath}. Skipping.")
                        continue
                    
                    # Standardize the ID key for the rest of the script to use.
                    item['id'] = unique_id
                    # --- MODIFIED SECTION END ---

                    # Ensure each node has a place to store its children
                    item['children'] = []
                    nodes[item['id']] = item
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line {line_num} in {input_filepath}. Skipping.")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    if not nodes:
        print(f"Warning: No valid data found in {input_filepath}")
        return

    # --- Step 2: Build the tree structure by linking children to parents ---
    root_nodes = []
    all_child_ids = set()

    # A second pass is needed to link children to their parents
    for node_id, node_data in nodes.items():
        # The parent_id for a comment is prefixed with 't1_' and for a post with 't3_'.
        # We need the raw ID to find it in our `nodes` dictionary.
        parent_id_full = node_data.get('parent_id')
        if parent_id_full:
            # Extract the raw ID (e.g., 't1_abcde' -> 'abcde')
            raw_parent_id = parent_id_full.split('_')[-1]
            
            if raw_parent_id in nodes:
                # This is a child node; add it to its parent's children list
                parent_node = nodes[raw_parent_id]
                parent_node['children'].append(node_data)
                all_child_ids.add(node_id)

    # Any node that is not a child of another node in the dataset is a root post
    for node_id, node_data in nodes.items():
        if node_id not in all_child_ids:
            root_nodes.append(node_data)

    # --- Step 3: Traverse each tree to add depth and sort children ---
    def process_node(node, depth):
        """Recursively add depth and sort children by timestamp."""
        node['depth'] = depth
        
        # Sort children by creation time to maintain conversation flow
        if node['children']:
            node['children'].sort(key=lambda x: x.get('created_utc', 0))
            for child in node['children']:
                process_node(child, depth + 1)

    for root in root_nodes:
        process_node(root, 0) # The post itself is at depth 0

    # --- Step 4: Write the completed trees to the output file ---
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for root in root_nodes:
                f.write(json.dumps(root) + '\n')
        print(f"Successfully created conversation trees at: {output_filepath}")

    except IOError as e:
        print(f"Error writing to output file {output_filepath}: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    # Define the directories where your data is and where you want to save the networks
    input_data_dir = 'data/scored/'
    output_network_dir = 'data/networks/'

    # Create dummy data for demonstration if the input directory doesn't exist
    if not os.path.exists(input_data_dir):
        print("Creating dummy data for demonstration...")
        os.makedirs(input_data_dir)
        dummy_filepath = os.path.join(input_data_dir, 'subreddit_askscience.jsonl')
        with open(dummy_filepath, 'w', encoding='utf-8') as f:
            # Post (root of the tree)
            f.write(json.dumps({"id": "post1", "post_id": "post1", "record_type": "post", "parent_id": "t5_2qm4e", "body": "Why is the sky blue?", "toxicity": 0.05, "created_utc": 1672531200}) + '\n')
            # Comment replying to post
            f.write(json.dumps({"id": "comment1", "comment_id": "comment1", "record_type": "comment", "parent_id": "t3_post1", "body": "It's due to Rayleigh scattering.", "toxicity": 0.02, "created_utc": 1672531300}) + '\n')
            # Another comment replying to post
            f.write(json.dumps({"id": "comment2", "comment_id": "comment2", "record_type": "comment", "parent_id": "t3_post1", "body": "Good question!", "toxicity": 0.01, "created_utc": 1672531250}) + '\n')
            # A reply to the first comment
            f.write(json.dumps({"id": "comment3", "comment_id": "comment3", "record_type": "comment", "parent_id": "t1_comment1", "body": "Could you explain scattering?", "toxicity": 0.03, "created_utc": 1672531400}) + '\n')
            # A second, entirely separate post in the same subreddit
            f.write(json.dumps({"id": "post2", "post_id": "post2", "record_type": "post", "parent_id": "t5_2qm4e", "body": "What is the speed of light?", "toxicity": 0.1, "created_utc": 1672617600}) + '\n')

    # Ensure the main output directory exists
    os.makedirs(output_network_dir, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_data_dir):
        if filename.endswith('.jsonl'):
            subreddit_name = filename.replace('.jsonl', '').replace('subreddit_', '')
            print(f"\nProcessing subreddit: {subreddit_name}...")
            
            input_file = os.path.join(input_data_dir, filename)
            output_file = os.path.join(output_network_dir, f'tree_{subreddit_name}.jsonl')
            
            build_conversation_trees(input_file, output_file)