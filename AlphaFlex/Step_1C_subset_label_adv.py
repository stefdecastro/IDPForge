import json
import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found.")
    sys.exit(1)

def main(args):
    min_p, max_p = args.min_len, args.max_len
    idr_len_range = (args.idr_min_len, args.idr_max_len)
    idr_len_range_str = f"{idr_len_range[0]}-{idr_len_range[1]}"

    print(f"--- Step 1D: Generating Subset ({min_p}-{max_p}), Report, Histogram ---")
    print(f"Matching Criteria:")
    print(f"  Protein Length: {min_p} <= length <= {max_p}")
    print(f"  IDR Count: Exactly {args.idr_count}")
    print(f"  IDR Type: {args.idr_type}")
    print(f"  IDR Length Range: {idr_len_range_str}")

    # 1. Load Databases
    try:
        with open(args.labeled_db, 'r') as f:
            master_db = json.load(f)
        with open(args.length_ref, 'r') as f:
            num_residues_db = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading database files: {e}")
        return

    report_data = []
    
    # 2. Process Data
    for prot_id, data in master_db.items():
        
        # --- A. Protein Length Filter (Dynamic) ---
        total_prot_len = num_residues_db.get(prot_id, 0)
        
        # LOGIC CHANGE: 
        # Use strictly inclusive logic (min <= x <= max).
        # To split 0-250 and 250-500 without overlap, run:
        # Run 1: --min_len 0   --max_len 250
        # Run 2: --min_len 251 --max_len 500
        if not (min_p <= total_prot_len <= max_p):
            continue

        # --- B. Strict IDR Count Filter ---
        labeled_idrs = data.get('labeled_idrs', [])
        if args.idr_count is not None:
            if len(labeled_idrs) != args.idr_count:
                continue

        # --- C. Type and IDR Length Filter ---
        match_details = None
        for idr in labeled_idrs:
            type_ok = (args.idr_type is None or
                       idr['type'] == args.idr_type)

            start, end = idr['range']
            idr_len = (end - start) + 1
            len_ok = True

            if idr_len_range:
                min_i, max_i = idr_len_range
                if not (min_i <= idr_len <= max_i):
                    len_ok = False
            
            if type_ok and len_ok:
                match_details = {
                    "idr_type": idr['type'],
                    "range_str": f"{start}-{end}",
                    "idr_len": idr_len
                }
                break
        
        if match_details:
            report_data.append({
                "prot_id": prot_id,
                "total_len": total_prot_len,
                "match_info": match_details
            })

    # --- Sampling ---
    if args.max_samples and len(report_data) > args.max_samples:
        report_data = random.sample(report_data, args.max_samples)
    
    report_data.sort(key=lambda x: x["prot_id"])
    
    id_list = [entry["prot_id"] for entry in report_data]
    length_distribution = [entry["match_info"]["idr_len"] for entry in report_data]
    
    # Setup Output Directory
    output_dir = os.path.join(args.output_root, "custom_subsets")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- NEW: Dynamic Filenames based on Range ---
    # This prevents overwriting when you run different ranges
    file_tag = f"{min_p}-{max_p}_{idr_len_range_str.replace('-', '_')}"
    
    list_path = os.path.join(output_dir, f"single_tail_subset_{file_tag}.txt")
    report_path = os.path.join(output_dir, f"filtered_report_{file_tag}.txt")
    hist_path = os.path.join(output_dir, f"idr_length_histogram_{file_tag}.png")
    table_path = os.path.join(output_dir, f"histogram_table_{file_tag}.txt")

    # =========================================================================
    # OUTPUT 1: The Original ID List
    # =========================================================================
    with open(list_path, 'w') as f:
        f.write("\n".join(id_list))

    # =========================================================================
    # OUTPUT 2: The Table Report
    # =========================================================================
    with open(report_path, 'w') as f:
        f.write(f" FILTRATION REPORT (Length {min_p} - {max_p}, IDR Length {idr_len_range_str})\n")
        f.write(f" Total Matches: {len(report_data)}\n")
        f.write("="*80 + "\n")
        header = f"{'Protein ID':<15} | {'Total Len':<10} | {'IDR Type':<15} | {'Range':<15} | {'IDR Len':<10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for entry in report_data:
            m = entry['match_info']
            row = f"{entry['prot_id']:<15} | {entry['total_len']:<10} | {m['idr_type']:<15} | {m['range_str']:<15} | {m['idr_len']:<10}"
            f.write(row + "\n")

    if length_distribution:
        max_val = max(length_distribution)
        bins = list(range(0, max_val + 100, 50))
        
        # OUTPUT 3: Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(length_distribution, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"IDR Lengths (Prot Len {min_p}-{max_p}, IDR Len {idr_len_range_str})")
        plt.xlabel("IDR Length")
        plt.ylabel("Frequency")
        plt.xticks(bins) 
        plt.grid(axis='y', alpha=0.5)
        plt.savefig(hist_path)
        plt.close()
        
        # OUTPUT 4: Data Table
        counts, bin_edges = np.histogram(length_distribution, bins=bins)
        with open(table_path, 'w') as f:
            f.write(f" HISTOGRAM DATA ({min_p}-{max_p}, IDR Len {idr_len_range_str})\n")
            f.write("-" * 28 + "\n")
            for i in range(len(counts)):
                range_str = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                f.write(f"{range_str:<15} | {counts[i]:<10}\n")

    print("-" * 50)
    print(f"PROCESSING COMPLETE ({min_p}-{max_p})")
    print(f"Saved list to: {list_path}")
    print(f"Total Matches: {len(report_data)}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1C: Advanced Subset Filtering & Histogram Generation")
    parser.add_argument("--labeled_db", default=cfg.LABELED_DB_PATH,
                        help=f"Path to labeled database JSON (default: {cfg.LABELED_DB_PATH}).")
    parser.add_argument("--length_ref", default=cfg.LENGTH_REF_PATH,
                        help=f"Path to residue-count reference JSON (default: {cfg.LENGTH_REF_PATH}).")
    parser.add_argument("--output_root", default=cfg.ID_LISTS_OUTPUT_ROOT,
                        help=f"Root directory for output files (default: {cfg.ID_LISTS_OUTPUT_ROOT}).")
    parser.add_argument("--min_len", type=int, default=0,
                        help="Minimum protein length, inclusive (default: 0).")
    parser.add_argument("--max_len", type=int, default=500,
                        help="Maximum protein length, inclusive (default: 500).")
    parser.add_argument("--idr_count", type=int, default=1,
                        help="Required number of IDRs per protein (default: 1).")
    parser.add_argument("--idr_type", default="Tail IDR",
                        help="Required IDR type, e.g. 'Tail IDR', 'Linker IDR' (default: 'Tail IDR').")
    parser.add_argument("--idr_min_len", type=int, default=0,
                        help="Minimum IDR length, inclusive (default: 0).")
    parser.add_argument("--idr_max_len", type=int, default=75,
                        help="Maximum IDR length, inclusive (default: 75).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum output proteins; None for no limit (default: None).")
    args = parser.parse_args()
    main(args)