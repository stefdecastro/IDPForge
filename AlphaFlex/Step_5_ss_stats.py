import os
import sys
import glob
import json
import argparse
import tempfile
import numpy as np
import mdtraj as md
import pandas as pd
from tqdm import tqdm

# --- 1. Import Auto Config ---
try:
    import config as cfg
except ImportError:
    print("Warning: auto_config.py not found. Relying on manual arguments.")
    cfg = None

# --- 2. Helper Functions ---

def get_idr_ranges(protein_id, labeled_db):
    """Extracts IDR ranges (start, end) from the DB."""
    if protein_id not in labeled_db:
        return []
    entry = labeled_db[protein_id]
    ranges = []
    if 'labeled_idrs' in entry:
        for item in entry['labeled_idrs']:
            if item['type'] != 'Folded Domain': 
                start, end = item['range']
                ranges.append((start, end))
    return ranges

def calculate_combined_stats(phi_array, psi_array, dssp_array):
    """
    Calculates both Ramachandran (Torsion) and DSSP fractions.
    """
    total_res = phi_array.size
    if total_res == 0:
        return None

    # --- A. TORSION STATS (Ramachandran) ---
    # Alpha: -180 < phi < 10  AND  -120 < psi < 45
    alpha_mask = (
        (phi_array > -180) & (phi_array < 10) & 
        (psi_array > -120) & (psi_array < 45)
    )
    # Beta: (-180 < phi < 0) AND ( (-180 < psi < -120) OR (45 < psi < 180) )
    beta_phi_cond = (phi_array > -180) & (phi_array < 0)
    beta_psi_cond = (
        ((psi_array > -180) & (psi_array < -120)) | 
        ((psi_array > 45) & (psi_array < 180))
    )
    beta_mask = beta_phi_cond & beta_psi_cond

    rama_alpha = np.sum(alpha_mask) / total_res
    rama_beta = np.sum(beta_mask) / total_res
    rama_other = 1.0 - rama_alpha - rama_beta

    # --- B. DSSP STATS (Secondary Structure) ---
    unique, counts = np.unique(dssp_array, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    dssp_h = counts_dict.get('H', 0) / total_res
    dssp_e = counts_dict.get('E', 0) / total_res
    dssp_c = counts_dict.get('C', 0) / total_res
    
    return {
        # Ramachandran (Geometric)
        "Rama_Alpha": rama_alpha,
        "Rama_Beta": rama_beta,
        "Rama_Other": rama_other,
        # DSSP (Hydrogen Bonding)
        "DSSP_H": dssp_h,
        "DSSP_E": dssp_e,
        "DSSP_C": dssp_c
    }

def yield_pdb_models(pdb_path):
    """
    Splitter Mode: Reads a multi-model PDB file textually and yields temp files.
    """
    with open(pdb_path, 'r') as f_in:
        lines = f_in.readlines()

    current_model_lines = []
    in_model = False
    found_any_model = False

    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            found_any_model = True
            current_model_lines = [] 
        elif line.startswith("ENDMDL"):
            in_model = False
            if current_model_lines:
                with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp:
                    tmp.writelines(current_model_lines)
                    tmp_path = tmp.name
                yield tmp_path
                if os.path.exists(tmp_path): os.remove(tmp_path)
            current_model_lines = []
        else:
            if line.startswith(("ATOM", "HETATM", "TER", "CONECT")):
                current_model_lines.append(line)

    if not found_any_model and current_model_lines:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp:
            tmp.writelines(current_model_lines)
            tmp_path = tmp.name
        yield tmp_path
        if os.path.exists(tmp_path): os.remove(tmp_path)

def process_ensemble_structure(pdb_input, protein_id, idr_ranges):
    if isinstance(pdb_input, list): pdb_input = pdb_input[0]
    
    files_to_process = [pdb_input]

    # --- 1. Initial Load Check ---
    try:
        test_load = md.load(pdb_input)
        del test_load
    except Exception as e:
        err_msg = str(e)
        if "contain the same number of ATOMs" in err_msg or "topology" in err_msg.lower():
            tqdm.write(f"  [WARN] {protein_id}: Atom count mismatch. Switching to Splitter Mode.")
            files_to_process = yield_pdb_models(pdb_input)
        else:
            tqdm.write(f"  [ERROR] {protein_id}: Critical Load Error ({e})")
            return None

    agg_phi = []
    agg_psi = []
    agg_dssp = []
    
    # --- 2. Iterative Processing ---
    for current_pdb in files_to_process:
        try:
            traj = md.load(current_pdb)
            n_res_total = traj.n_residues
            n_frames = traj.n_frames

            # --- A. Compute Torsions ---
            idx_phi, vals_phi = md.compute_phi(traj)
            idx_psi, vals_psi = md.compute_psi(traj)
            
            res_indices_phi = [traj.topology.atom(atoms[1]).residue.index for atoms in idx_phi]
            res_indices_psi = [traj.topology.atom(atoms[1]).residue.index for atoms in idx_psi]

            frame_phi = np.full((n_frames, n_res_total), np.nan)
            frame_psi = np.full((n_frames, n_res_total), np.nan)

            frame_phi[:, res_indices_phi] = np.rad2deg(vals_phi)
            frame_psi[:, res_indices_psi] = np.rad2deg(vals_psi)

            # --- B. Compute DSSP ---
            frame_dssp = np.full((n_frames, n_res_total), 'NA', dtype='<U2')
            try:
                dssp_codes = md.compute_dssp(traj, simplified=True)
                if dssp_codes.shape == frame_dssp.shape:
                    frame_dssp = dssp_codes
                else:
                    tqdm.write(f"  [WARN] {protein_id}: DSSP shape mismatch.")
            except Exception:
                pass

            # --- C. Slicing (Chain Aware) ---
            for chain in traj.topology.chains:
                chain_map = {res.resSeq: res.index for res in chain.residues}
                if not chain_map: continue
                
                for start, end in idr_ranges:
                    valid_indices = []
                    for r_num in range(start, end + 1):
                        if r_num in chain_map:
                            valid_indices.append(chain_map[r_num])
                    
                    if valid_indices:
                        idx_array = np.array(valid_indices)
                        agg_phi.append(frame_phi[:, idx_array])
                        agg_psi.append(frame_psi[:, idx_array])
                        agg_dssp.append(frame_dssp[:, idx_array])
            del traj

        except Exception as e:
            tqdm.write(f"  [ERROR] {protein_id}: Crash inside loop -> {e}")
            continue

    if not agg_phi:
        tqdm.write(f"  [FAIL] {protein_id}: No valid IDR data extracted.")
        return None

    # --- 3. Flatten & Calculate ---
    combined_phi = np.concatenate([x.flatten() for x in agg_phi])
    combined_psi = np.concatenate([x.flatten() for x in agg_psi])
    combined_dssp = np.concatenate([x.flatten() for x in agg_dssp])
    
    valid_mask = (~np.isnan(combined_phi)) & (~np.isnan(combined_psi))
    
    clean_phi = combined_phi[valid_mask]
    clean_psi = combined_psi[valid_mask]
    clean_dssp = combined_dssp[valid_mask]
    
    return calculate_combined_stats(clean_phi, clean_psi, clean_dssp)

# --- 3. Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Batch calculate IDR Structure (Torsion + DSSP) stats.")
    parser.add_argument("--db", default=None, help="Path to labeled_db.json.")
    parser.add_argument("--input_dir", default=None, help="Manually specify a directory to scan.")
    parser.add_argument("--output_csv", default="per_protein_structure_stats.csv", help="Output file.")
    parser.add_argument("--output_summary", default="total_structure_report.txt", help="Output report.")
    args = parser.parse_args()

    # --- Resolve DB Path ---
    db_path = args.db
    if db_path is None:
        if cfg and hasattr(cfg, 'LABELED_DB_PATH'):
            db_path = cfg.LABELED_DB_PATH
        else:
            print("Error: LABELED_DB_PATH not found in config.")
            sys.exit(1)

    with open(db_path, 'r') as f:
        labeled_db = json.load(f)

    # --- Resolve Scan Directories & Bin Name ---
    dirs_to_scan = []
    bin_name = "Unknown_Bin"
    categories = [
        "cat_0_idp_only", "cat_1_tail_only", "cat_2_linker_only", "cat_3_loop_only",
        "cat_1_and_2_mixed", "cat_2_and_3_mixed", "cat_1_and_3_mixed", "cat_1_2_3_mixed"
    ]

    if args.input_dir:
        if os.path.isdir(args.input_dir):
            print(f"Scanning manual input: {args.input_dir}")
            dirs_to_scan.append(args.input_dir)
            bin_name = os.path.basename(args.input_dir.rstrip(os.sep))
        else:
            sys.exit(1)
    else:
        base_dir = None
        if cfg and hasattr(cfg, 'STITCH_OUTPUT_ROOT') and os.path.exists(cfg.STITCH_OUTPUT_ROOT):
            base_dir = cfg.STITCH_OUTPUT_ROOT
            print(f"Auto-detected Step 4 Dir: {base_dir}")
        elif cfg and hasattr(cfg, 'CONFORMER_POOL_DIR') and os.path.exists(cfg.CONFORMER_POOL_DIR):
            base_dir = cfg.CONFORMER_POOL_DIR
            print(f"Auto-detected Step 3 Dir: {base_dir}")
        
        if base_dir:
            bin_name = os.path.basename(base_dir.rstrip(os.sep))
            for cat in categories:
                target_path = os.path.join(base_dir, cat, "Complete")
                if os.path.exists(target_path):
                    print(f"  [FOUND] {cat}/Complete")
                    dirs_to_scan.append(target_path)

    if not dirs_to_scan:
        print("Error: No directories found.")
        sys.exit(0)

    # --- Processing ---
    stats_list = []
    
    for current_dir in dirs_to_scan:
        print(f"\nProcessing: {current_dir}")
        items = os.listdir(current_dir)
        
        for item in tqdm(items, desc="Proteins"):
            path = os.path.join(current_dir, item)
            protein_id = None
            pdb_file = None

            if os.path.isdir(path):
                protein_id = item
                sub_path = os.path.join(path, "stitched_ensemble")
                search_path = sub_path if os.path.isdir(sub_path) else path
                
                ensembles = glob.glob(os.path.join(search_path, "*_ensemble_n*.pdb"))
                if ensembles: pdb_file = ensembles[0]
                else:
                    singles = glob.glob(os.path.join(search_path, "*_conformer_*.pdb"))
                    if singles: pdb_file = singles 
            
            elif item.endswith(".pdb"):
                protein_id = item.split("_")[0]
                pdb_file = path

            if not protein_id or not pdb_file: continue

            ranges = get_idr_ranges(protein_id, labeled_db)
            if not ranges: continue
            
            stats = process_ensemble_structure(pdb_file, protein_id, ranges)
            
            if stats:
                stats["protein_id"] = protein_id
                stats["category"] = os.path.basename(os.path.dirname(current_dir)) 
                stats_list.append(stats)

    # --- Reporting ---
    if not stats_list:
        print("\nNo valid IDR data extracted.")
        sys.exit(0)

    df = pd.DataFrame(stats_list)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved stats to {args.output_csv}")

    # Statistics Calculation
    metrics = ["Rama_Alpha", "Rama_Beta", "Rama_Other", "DSSP_H", "DSSP_E", "DSSP_C"]
    means = df[metrics].mean() * 100
    stds = df[metrics].std() * 100
    mins = df[metrics].min() * 100
    maxs = df[metrics].max() * 100

    report = f"""
==================================================
           GLOBAL IDR STRUCTURE REPORT
==================================================
Bin / Dataset:    {bin_name}
Total Proteins:   {len(df)}
Database Used:    {db_path}

--- Ramachandran (Backbone Geometry) ---
Alpha (Helix-like):  {means['Rama_Alpha']:.2f}% (± {stds['Rama_Alpha']:.2f}%) [Range: {mins['Rama_Alpha']:.2f}% - {maxs['Rama_Alpha']:.2f}%]
Beta  (Sheet/PPII):  {means['Rama_Beta']:.2f}%  (± {stds['Rama_Beta']:.2f}%)  [Range: {mins['Rama_Beta']:.2f}% - {maxs['Rama_Beta']:.2f}%]
Other (Disordered):  {means['Rama_Other']:.2f}% (± {stds['Rama_Other']:.2f}%) [Range: {mins['Rama_Other']:.2f}% - {maxs['Rama_Other']:.2f}%]

--- DSSP (Hydrogen Bonding) ---
Helix (H):           {means['DSSP_H']:.2f}% (± {stds['DSSP_H']:.2f}%) [Range: {mins['DSSP_H']:.2f}% - {maxs['DSSP_H']:.2f}%]
Sheet (E):           {means['DSSP_E']:.2f}% (± {stds['DSSP_E']:.2f}%) [Range: {mins['DSSP_E']:.2f}% - {maxs['DSSP_E']:.2f}%]
Coil/Loop (C):       {means['DSSP_C']:.2f}% (± {stds['DSSP_C']:.2f}%) [Range: {mins['DSSP_C']:.2f}% - {maxs['DSSP_C']:.2f}%]

--- Output File ---
{args.output_csv}
"""
    print(report)
    with open(args.output_summary, "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()