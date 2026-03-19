"""
Step 1B: Subset Sampling & ID List Generation
=============================================

Creates lists of protein IDs filtered by length and groups them into the 4 mutually exclusive 
categories defined in Step 1:
  - Cat 0: Full IDP
  - Cat 1: Tails (only)
  - Cat 2: Linkers (may have tails)
  - Cat 3: Loops (may have linkers/tails)

Output:
  data/id_lists/0-1000_AA/
     ├── cat_0.txt
     ├── cat_1.txt
     ├── cat_2.txt
     └── cat_3.txt
"""

import json
import os
import random
import sys
import argparse
from collections import defaultdict

try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'project_config.py' not found.")
    sys.exit(1)

def main(args):
    def log(msg, force=False):
        if args.verbose or force:
            print(msg)

    log("--- Step 1B: Subset Generation ---", force=True)

    if not os.path.exists(args.labeled_db):
        print(f"ERROR: Labeled DB not found: {args.labeled_db}")
        sys.exit(1)

    log(f"Loading labeled DB...")
    with open(args.labeled_db, 'r') as f:
        master_db = json.load(f)

    log(f"Loading length reference...")
    with open(args.length_ref, 'r') as f:
        num_residues_db = json.load(f)

    # Output Setup
    range_label = f"{args.min_length}-{args.max_length}AA"
    output_dir = os.path.join(args.output_root, range_label)
    os.makedirs(output_dir, exist_ok=True)

    log(f"Target Range: {args.min_length} - {args.max_length} residues", force=True)
    log(f"Output Folder: {output_dir}", force=True)

    pools = defaultdict(list)
    count_skipped_len = 0
    count_skipped_no_cat = 0

    # Filtering
    for prot_id, data in master_db.items():
        length = num_residues_db.get(prot_id, 0)
        
        # Inclusive bounds check
        if not (args.min_length <= length <= args.max_length):
            count_skipped_len += 1
            continue

        cat = data.get('category')
        if cat is None or cat == -1:
            count_skipped_no_cat += 1
            continue
            
        pools[cat].append(prot_id)

    # Sampling
    log("\n--- Generating Lists ---", force=True)
    summary_counts = {}

    for cat_id in [0, 1, 2, 3]:
        id_list = pools[cat_id]
        total_available = len(id_list)
        
        filename = f"cat_{cat_id}.txt"
        filepath = os.path.join(output_dir, filename)

        if total_available == 0:
            log(f"  [ ] {filename:<12} : Empty (0 proteins)", force=True)
            open(filepath, 'w').close()
            continue
            
        sample_size = min(total_available, args.sample_size)
        sampled_ids = random.sample(id_list, sample_size)
        sampled_ids.sort() 
        
        try:
            with open(filepath, 'w') as f:
                f.write("\n".join(sampled_ids))
            
            # Always print summary of files created
            log(f"  [\u2713] {filename:<12} : {sample_size} IDs (Pool: {total_available})", force=True)
            summary_counts[f"cat_{cat_id}"] = sample_size
            
        except IOError as e:
            print(f"  [!] Error writing {filename}: {e}")

    # Summary
    summary_json_path = os.path.join(output_dir, "batch_manifest.json")
    with open(summary_json_path, 'w') as f:
        json.dump({
            "length_range": range_label,
            "min_len": args.min_length,
            "max_len": args.max_length,
            "counts": summary_counts,
        }, f, indent=4)

    log("\n--- Step 1B Complete ---", force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1B: Subset Sampling & ID List Generation")
    parser.add_argument("--labeled_db", default=cfg.LABELED_DB_PATH,
                        help=f"Path to labeled database JSON (default: {cfg.LABELED_DB_PATH}).")
    parser.add_argument("--length_ref", default=cfg.LENGTH_REF_PATH,
                        help=f"Path to residue-count reference JSON (default: {cfg.LENGTH_REF_PATH}).")
    parser.add_argument("--output_root", default=cfg.ID_LISTS_OUTPUT_ROOT,
                        help=f"Root directory for output ID lists (default: {cfg.ID_LISTS_OUTPUT_ROOT}).")
    parser.add_argument("--min_length", type=int, default=cfg.SUBSET_MIN_LENGTH,
                        help=f"Minimum protein length, inclusive (default: {cfg.SUBSET_MIN_LENGTH}).")
    parser.add_argument("--max_length", type=int, default=cfg.SUBSET_MAX_LENGTH,
                        help=f"Maximum protein length, inclusive (default: {cfg.SUBSET_MAX_LENGTH}).")
    parser.add_argument("--sample_size", type=int, default=cfg.SUBSET_SAMPLE_SIZE,
                        help=f"Max proteins per category (default: {cfg.SUBSET_SAMPLE_SIZE}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed logging.")
    args = parser.parse_args()
    main(args)