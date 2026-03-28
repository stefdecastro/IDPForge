"""
Step 3: Phased Conformer Generation Pipeline
1. Checks validated count.
2. If deficit, generates relaxed conformers via sample_ldr.py (includes minimization).
3. Phase 1: Chirality check on relaxed files (flip + re-relax if needed).
4. Phase 2: Validate all relaxed files with detailed per-conformer logging.
5. Renumbers successes to fill 1..N.
6. Repeats until target is met or max attempts exhausted.
"""
import os
import sys
import subprocess
import argparse
import yaml
import json
import time
import numpy as np
import glob
import re
import math
import gc
import torch
from openmm.app import PDBFile

# ------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    import config as cfg
    from utils.smart_scoring import get_smart_threshold
    from utils.pre_minimization import repair_chirality, fix_histidine_naming
    from utils.post_minimization import (validate_structure_post_relax,
                                         check_bond_integrity)
    from utils.file_ops import atomic_transfer

    from openfold.np import protein
    from idpforge.utils.relax import relax_protein
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Missing Dependency.\n{e}")

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
def load_relax_config():
    yaml_path = os.path.join(ROOT_DIR, "configs", "sample.yml")
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f).get('relax', {})
    except Exception:
        return {}

RELAX_CONFIG = load_relax_config()

SEP = "-" * 60
STATE_FILENAME = ".step3_state.json"


# ------------------------------------------------------------------------
# STATE PERSISTENCE (for cluster resumability)
# ------------------------------------------------------------------------
def _load_state(output_dir, verbose=False):
    """Load persisted state from a previous run, or return defaults."""
    state_path = os.path.join(output_dir, STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            if verbose:
                print(f"   [Resume] Loaded state: {state['total_attempts']} prior attempts.", flush=True)
            return state
        except Exception:
            pass
    return {"total_attempts": 0}


def _save_state(output_dir, total_attempts):
    """Persist state to disk so runs can be resumed."""
    state_path = os.path.join(output_dir, STATE_FILENAME)
    state = {"total_attempts": total_attempts}
    try:
        tmp = state_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, state_path)  # atomic on POSIX
    except Exception as e:
        print(f"   [Warning] Could not save state: {e}", flush=True)


# ------------------------------------------------------------------------
# ORPHAN RECOVERY (pick up files from a killed run)
# ------------------------------------------------------------------------
def _collect_orphaned_relaxed(output_dir, verbose=False):
    """Find *_relaxed.pdb files left over from a previous killed run."""
    relaxed = glob.glob(os.path.join(output_dir, "*_relaxed.pdb"))
    if relaxed:
        relaxed.sort(key=lambda p: _numeric_sort_key_relaxed(p))
        if verbose:
            print(f"   [Resume] Found {len(relaxed)} orphaned relaxed files.", flush=True)
    return relaxed


def _numeric_sort_key_relaxed(path):
    """Extract leading number from relaxed filename: 1_relaxed.pdb -> 1"""
    m = re.search(r'(\d+)_relaxed', os.path.basename(path))
    return int(m.group(1)) if m else 0


def _cleanup_dir(output_dir):
    """Remove leftover raw and relaxed files (but not validated or state)."""
    for pattern in ("*_raw.pdb", "*_relaxed.pdb", ".tmp_*"):
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


# ------------------------------------------------------------------------
# PHASE 1: GENERATION + RELAXATION (subprocess to sample_ldr.py)
# ------------------------------------------------------------------------
def generate_conformers(npz_path, output_dir, num_to_generate, verbose=False):
    """
    Calls sample_ldr.py (with relaxation) to produce N_relaxed.pdb files.
    HIS hydrogen stripping and AMBER minimization happen inside sample_ldr.
    Returns list of relaxed file paths created.
    """
    # Count existing relaxed files to determine starting counter
    existing = glob.glob(os.path.join(output_dir, "*_relaxed.pdb"))
    start_count = len(existing)
    target_total = start_count + num_to_generate

    if verbose:
        print(f"   [Gen] Generating {num_to_generate} new conformers (with relaxation)...", flush=True)

    cmd = [
        cfg.PYTHON_EXEC, cfg.SCRIPT_SAMPLE_LDR,
        cfg.MODEL_WEIGHTS_PATH, npz_path, output_dir, cfg.MODEL_CONFIG_PATH,
        "--batch", str(cfg.SAMPLE_BATCH_SIZE),
        "--nconf", str(target_total),
        "--ss_db", cfg.SS_DB_PATH
    ]
    if cfg.DEVICE == "cuda":
        cmd.append("--cuda")

    if verbose:
        print(f"   [Gen] Launching subprocess on GPU...", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run(cmd, check=True, env=env)

    # Clean up leftover raw files (sample_ldr writes raw then relaxes, but
    # does not delete the raw files itself)
    for raw_f in glob.glob(os.path.join(output_dir, "*_raw.pdb")):
        try:
            os.remove(raw_f)
        except OSError:
            pass

    # Collect all relaxed files
    relaxed_files = sorted(
        glob.glob(os.path.join(output_dir, "*_relaxed.pdb")),
        key=_numeric_sort_key_relaxed
    )
    if verbose:
        print(f"   [Gen] {len(relaxed_files)} relaxed files now in directory.", flush=True)
    return relaxed_files


# ------------------------------------------------------------------------
# PHASE 2: CHIRALITY CHECK & RE-RELAX (on minimized files)
# ------------------------------------------------------------------------
def _relax_single(pdb_path, output_dir, stem, mask, viol_mask, dev_idx):
    """
    Relax a single PDB file. Returns the relaxed path on success, None on failure.
    """
    import copy
    temp_relaxed_path = os.path.join(output_dir, f"{stem}_relaxed.pdb")

    try:
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()
        unrelaxed_prot = protein.from_pdb_string(pdb_str)

        relax_config = copy.deepcopy(RELAX_CONFIG)
        if mask is not None:
            relax_config["exclude_residues"] = np.where(~mask)[0].tolist()

        result = relax_protein(
            config=relax_config,
            model_device=dev_idx,
            unrelaxed_protein=unrelaxed_prot,
            output_dir=output_dir,
            pdb_name=stem,
            viol_mask=viol_mask
        )

        del unrelaxed_prot
        gc.collect()
        if cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

        if result == 1 and os.path.exists(temp_relaxed_path):
            return temp_relaxed_path
        else:
            if os.path.exists(temp_relaxed_path):
                os.remove(temp_relaxed_path)
            return None

    except Exception as e:
        print(f"      [Relax Error] {os.path.basename(pdb_path)}: {e}", flush=True)
        if os.path.exists(temp_relaxed_path):
            os.remove(temp_relaxed_path)
        return None


def repair_and_rerelax(relaxed_files, output_dir, mask, verbose=False):
    """
    Structural repair phase applied to every relaxed file:
      1. check_bond_integrity  -> identify broken HIS ring bonds
      2. repair_chirality      -> identify D-amino acids
      3. fix_histidine_naming  -> reassign HIS atom names (if flagged)
      4. re-relax              -> only if either fix was applied

    Returns a (possibly updated) list of relaxed file paths.
    """
    _HIS_RESNAMES = {'HIS', 'HID', 'HIE', 'HIP'}
    viol_mask = ~mask.astype(bool) if mask is not None else None
    dev_idx = "0" if cfg.DEVICE == "cuda" else "cpu"

    result_files = []
    total_chiral = 0
    total_his_fixed = 0

    for relaxed_path in relaxed_files:
        fname = os.path.basename(relaxed_path)
        needs_rerelax = False

        # --- 1. Bond integrity check -> find broken HIS residues ---
        try:
            chk_pdb = PDBFile(relaxed_path)
            broken = check_bond_integrity(chk_pdb.topology, chk_pdb.positions)
            his_resids = {b['resid'] for b in broken
                          if b['resname'] in _HIS_RESNAMES
                          or b.get('resname2', '') in _HIS_RESNAMES}
        except Exception as e:
            print(f"      [Bond Check Error] {fname}: {e}", flush=True)
            his_resids = set()

        # --- 2. Chirality check -> flip D-amino acids ---
        try:
            n_chiral = repair_chirality(relaxed_path, verbose=False)
        except Exception as e:
            print(f"      [Chirality Error] {fname}: {e}", flush=True)
            n_chiral = 0

        if n_chiral > 0:
            total_chiral += n_chiral
            needs_rerelax = True
            if verbose:
                print(f"       [REPAIR] Flipped {n_chiral} D-isomer(s) in {fname}.", flush=True)

        # --- 3. Fix HIS atom naming (only flagged residues) ---
        if his_resids:
            try:
                n_his = fix_histidine_naming(relaxed_path, his_resids, verbose=False)
                if n_his > 0:
                    total_his_fixed += n_his
                    needs_rerelax = True
            except Exception as e:
                print(f"       [ERROR] HIS naming fix failed for {fname}: {e}", flush=True)

        # --- 4. Re-relax if either fix was applied ---
        if not needs_rerelax:
            result_files.append(relaxed_path)
            continue

        if verbose:
            print(f"      [Re-relax] Re-relaxing {fname}...", flush=True)
        stem = fname.replace("_relaxed.pdb", "") + "_rerelax"
        rerelaxed_path = _relax_single(
            relaxed_path, output_dir, stem, mask, viol_mask, dev_idx
        )

        if rerelaxed_path is not None:
            os.replace(rerelaxed_path, relaxed_path)
            result_files.append(relaxed_path)
        else:
            if verbose:
                print(f"      [Re-relax] Failed for {fname}, discarding.", flush=True)
            if os.path.exists(relaxed_path):
                os.remove(relaxed_path)

    # --- Summary (always print) ---
    repairs = []
    if total_chiral > 0:
        repairs.append(f"{total_chiral} D-amino acid(s)")
    if total_his_fixed > 0:
        repairs.append(f"{total_his_fixed} HIS naming(s)")
    if repairs:
        print(f"   [REPAIR] Fixed {', '.join(repairs)}. "
              f"{len(result_files)}/{len(relaxed_files)} survived re-relaxation.",
              flush=True)
    else:
        print(f"   [REPAIR] All {len(relaxed_files)} files clean.", flush=True)

    return result_files


# ------------------------------------------------------------------------
# PHASE 3: VALIDATE (with detailed per-conformer logging)
# ------------------------------------------------------------------------
def validate_all(relaxed_files, output_dir, idr_range, total_attempts, num_validated, verbose=False):
    """
    Validates each relaxed file with detailed logging matching the requested format.
    Returns (num_new_valid, total_attempts_after).
    """
    target = cfg.SAMPLE_N_CONFS
    next_idx = num_validated + 1
    new_valid = 0

    for relaxed_path in relaxed_files:
        if num_validated + new_valid >= target:
            # Already reached target, clean up remaining
            os.remove(relaxed_path)
            continue

        total_attempts += 1
        threshold = get_smart_threshold(total_attempts, num_validated + new_valid)

        if verbose:
            print(SEP, flush=True)
            print(f"     [Attempt {total_attempts}] Validating {os.path.basename(relaxed_path)}...", flush=True)

        t0 = time.perf_counter()

        try:
            chk_pdb = PDBFile(relaxed_path)

            # Single call — full_report=True ensures all 4 checks run
            is_valid, info = validate_structure_post_relax(
                chk_pdb.topology, chk_pdb.positions,
                pdb_path=relaxed_path,
                strict_clash_threshold=threshold,
                idr_start=idr_range[0], idr_end=idr_range[1],
                verbose=False,
                full_report=True
            )

            elapsed = time.perf_counter() - t0

            if verbose:
                # --- Extract results for logging ---
                chiral_pass = info.get("chirality_pass", True)
                bonds_pass = info.get("bonds_pass", True)
                clash_pass = info.get("clash_pass", True)
                knot_pass = info.get("knot_pass", True)

                clash_count = info.get("num_clashes", "?")
                clash_score = info.get("clash_score", 0.0)
                n_broken = info.get("num_broken_bonds", 0)
                knot_type = info.get("knot_type", "None")

                chiral_str = "PASS" if chiral_pass else "FAIL (D-Amino detected)"
                bonds_str = "PASS" if bonds_pass else f"FAIL ({n_broken} broken)"
                clash_str = "PASS" if clash_pass else "FAIL"
                knot_str = "PASS" if knot_pass else f"FAIL ({knot_type})"

                # --- Print detailed log (matches validation order) ---
                print(f"       [TIMING] Validate: {elapsed:.2f}s", flush=True)
                print(f"       [POST-MIN CHECK] Validating...", flush=True)
                print(f"         - Chirality: {chiral_str}", flush=True)
                print(f"         - Bonds:     {bonds_str}", flush=True)
                print(f"         - Clashes:   {clash_count}  (Score: {clash_score:.2f} | Limit: {threshold:.1f}) -> {clash_str}", flush=True)
                print(f"         - Topology:  {knot_str}", flush=True)

            if is_valid:
                validated_name = f"{next_idx}_validated.pdb"
                atomic_transfer(relaxed_path, output_dir, validated_name)
                new_valid += 1
                count_display = num_validated + new_valid
                if verbose:
                    print(f"       [RESULT] SUCCESS! Count: {count_display}/{target}", flush=True)
                next_idx += 1
            else:
                if verbose:
                    print(f"       [RESULT] FAILED ({info['reason']}) [Thresh: {threshold:.2f}]", flush=True)

                # Delete failed relaxed file
                if os.path.exists(relaxed_path):
                    os.remove(relaxed_path)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"       [TIMING] {elapsed:.2f}s", flush=True)
            print(f"       [RESULT] CRASHED: {e}", flush=True)
            if os.path.exists(relaxed_path):
                os.remove(relaxed_path)

    if verbose:
        print(SEP, flush=True)
    return new_valid, total_attempts


# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------
def _count_validated(output_dir):
    """Count existing validated files and return (count, next_index)."""
    v_files = glob.glob(os.path.join(output_dir, "*_validated.pdb"))
    if not v_files:
        return 0, 1
    indices = []
    for f in v_files:
        m = re.search(r'(\d+)_validated', os.path.basename(f))
        if m:
            indices.append(int(m.group(1)))
    return len(v_files), (max(indices) + 1 if indices else 1)


# ------------------------------------------------------------------------
# MAIN WORKFLOW PER IDR
# ------------------------------------------------------------------------
def run_idr_workflow(prot_id, npz_path, start_res, end_res, verbose=False):
    range_tag = f"idr_{start_res}-{end_res}"
    output_dir = os.path.join(cfg.CONFORMER_POOL_DIR, prot_id, range_tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n   >>> Processing Region: {range_tag}", flush=True)

    try:
        dat = np.load(npz_path)
        fixed_mask = dat['mask']
    except Exception:
        fixed_mask = None

    idr_range = (start_res, end_res)

    # Load persisted state (survives cluster preemption / restarts)
    state = _load_state(output_dir, verbose=verbose)
    total_attempts = state["total_attempts"]

    # ------------------------------------------------------------------
    # RESUME PHASE: Process orphaned relaxed files from a killed run
    # ------------------------------------------------------------------
    orphaned_relaxed = _collect_orphaned_relaxed(output_dir, verbose=verbose)
    if orphaned_relaxed:
        num_val, _ = _count_validated(output_dir)

        if verbose:
            print(f"\n   --- Resume Phase: Chirality ({len(orphaned_relaxed)} orphaned relaxed) ---", flush=True)
        orphaned_relaxed = repair_and_rerelax(orphaned_relaxed, output_dir, fixed_mask, verbose=verbose)

        if orphaned_relaxed:
            if verbose:
                print(f"\n   --- Resume Phase: Validate ({len(orphaned_relaxed)} orphaned relaxed) ---", flush=True)
            new_valid, total_attempts = validate_all(
                orphaned_relaxed, output_dir, idr_range, total_attempts, num_val, verbose=verbose
            )
            _save_state(output_dir, total_attempts)
            if verbose:
                print(f"   [Resume Summary] +{new_valid} recovered from orphaned relaxed files.", flush=True)

    # Clean up any leftover raw files (sample_ldr may have left them)
    for raw_f in glob.glob(os.path.join(output_dir, "*_raw.pdb")):
        try:
            os.remove(raw_f)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # MAIN LOOP: Generate+Relax -> Chirality Check/Re-relax -> Validate
    # ------------------------------------------------------------------
    while total_attempts < cfg.SAMPLE_MAX_TOTAL_ATTEMPTS:
        # 1. Check current validated count
        num_val, next_idx = _count_validated(output_dir)

        if num_val >= cfg.SAMPLE_N_CONFS:
            print(f"   [Done] Target reached ({num_val}/{cfg.SAMPLE_N_CONFS}).", flush=True)
            _cleanup_dir(output_dir)
            break

        needed = cfg.SAMPLE_N_CONFS - num_val
        num_to_generate = max(math.ceil(needed * 1.2), 10)
        if verbose:
            print(f"   [Status] Have {num_val}/{cfg.SAMPLE_N_CONFS}. Need {needed}. Generating {num_to_generate}.", flush=True)

        # 2. PHASE 1: Generate + relax conformers (sample_ldr handles both)
        relaxed_files = generate_conformers(npz_path, output_dir, num_to_generate, verbose=verbose)

        if not relaxed_files:
            print(f"   [Warning] No relaxed files produced. Retrying...", flush=True)
            continue

        # 3. PHASE 2: Chirality check on relaxed files (flip + re-relax if needed)
        if verbose:
            print(f"\n   --- Phase: Chirality ({len(relaxed_files)} files) ---", flush=True)
        relaxed_files = repair_and_rerelax(relaxed_files, output_dir, fixed_mask, verbose=verbose)

        if not relaxed_files:
            print(f"   [Warning] No structures survived chirality repair. Retrying...", flush=True)
            continue

        # 4. PHASE 3: Validate all with detailed logging
        if verbose:
            print(f"\n   --- Phase: Validate ({len(relaxed_files)} files) ---", flush=True)
        new_valid, total_attempts = validate_all(
            relaxed_files, output_dir, idr_range, total_attempts, num_val, verbose=verbose
        )

        # Persist state after every round
        _save_state(output_dir, total_attempts)

        if verbose:
            print(f"\n   [Round Summary] +{new_valid} validated this round. "
                  f"Total attempts: {total_attempts}.", flush=True)

    else:
        print(f"   [ABORT] Max attempts ({cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}) reached "
              f"for {range_tag}.", flush=True)


# ------------------------------------------------------------------------
# TEMPLATE DISCOVERY
# ------------------------------------------------------------------------
def find_and_sort_templates(prot_id):
    search_pattern = f"{prot_id}_idr_*-*.npz"
    candidates = glob.glob(os.path.join(cfg.TEMPLATE_OUTPUT_DIR, "**", search_pattern), recursive=True)
    templates = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        fname = os.path.basename(path)
        match = re.search(r"_idr_(\d+)-(\d+)\.npz", fname)
        if match:
            templates.append({"path": path, "start": int(match.group(1)), "end": int(match.group(2))})
    templates.sort(key=lambda x: x["start"])
    return templates


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------
def main(args):
    verbose = args.verbose

    print("=" * 60, flush=True)
    print("  Phased Conformer Generation Pipeline (Step 3)", flush=True)
    print("=" * 60, flush=True)

    # Override config with CLI args so all helper functions pick them up
    cfg.TEMPLATE_OUTPUT_DIR = args.template_dir
    cfg.CONFORMER_POOL_DIR = args.output_dir
    cfg.SAMPLE_N_CONFS = args.n_confs
    cfg.SAMPLE_MAX_TOTAL_ATTEMPTS = args.max_attempts
    cfg.SAMPLE_BATCH_SIZE = args.batch_size
    cfg.DEVICE = args.device
    cfg.MODEL_WEIGHTS_PATH = args.weights
    cfg.MODEL_CONFIG_PATH = args.model_config
    cfg.SS_DB_PATH = args.ss_db

    with open(args.id_file, 'r') as f:
        all_ids = sorted({l.strip() for l in f if l.strip()})

    my_chunk = all_ids
    if args.total_splits > 1:
        my_chunk = np.array_split(all_ids, args.total_splits)[args.split_index].tolist()

    print(f"Job {args.split_index+1}/{args.total_splits}: Processing {len(my_chunk)} Proteins", flush=True)
    print(f"Target: {cfg.SAMPLE_N_CONFS} conformers per IDR | "
          f"Max attempts: {cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}", flush=True)

    for i, prot_id in enumerate(my_chunk):
        print(f"\n[{i+1}/{len(my_chunk)}] Protein: {prot_id}")
        templates = find_and_sort_templates(prot_id)

        if not templates:
            print(f"   [SKIP] No templates found.", flush=True)
            continue

        print(f"   Found {len(templates)} regions.", flush=True)
        for t in templates:
            run_idr_workflow(prot_id, t["path"], t["start"], t["end"], verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Phased Conformer Generation Pipeline")
    parser.add_argument("--id_file", required=True,
                        help="Path to a text file containing newline-separated UniProt IDs.")
    parser.add_argument("--total_splits", type=int, default=1,
                        help="Total number of parallel jobs (default: 1).")
    parser.add_argument("--split_index", type=int, default=0,
                        help="The specific shard index, 0-based (default: 0).")
    parser.add_argument("--template_dir", default=cfg.TEMPLATE_OUTPUT_DIR,
                        help=f"Directory containing Step 2 .npz templates (default: {cfg.TEMPLATE_OUTPUT_DIR}).")
    parser.add_argument("--output_dir", default=cfg.CONFORMER_POOL_DIR,
                        help=f"Output directory for generated conformers (default: {cfg.CONFORMER_POOL_DIR}).")
    parser.add_argument("--n_confs", type=int, default=cfg.SAMPLE_N_CONFS,
                        help=f"Target number of validated conformers per IDR (default: {cfg.SAMPLE_N_CONFS}).")
    parser.add_argument("--max_attempts", type=int, default=cfg.SAMPLE_MAX_TOTAL_ATTEMPTS,
                        help=f"Maximum total generation attempts per IDR (default: {cfg.SAMPLE_MAX_TOTAL_ATTEMPTS}).")
    parser.add_argument("--batch_size", type=int, default=cfg.SAMPLE_BATCH_SIZE,
                        help=f"Batch size for diffusion sampling (default: {cfg.SAMPLE_BATCH_SIZE}).")
    parser.add_argument("--device", default=cfg.DEVICE,
                        help=f"Device for inference: 'cuda' or 'cpu' (default: {cfg.DEVICE}).")
    parser.add_argument("--weights", default=cfg.MODEL_WEIGHTS_PATH,
                        help=f"Path to model weights checkpoint (default: {cfg.MODEL_WEIGHTS_PATH}).")
    parser.add_argument("--model_config", default=cfg.MODEL_CONFIG_PATH,
                        help=f"Path to model YAML config (default: {cfg.MODEL_CONFIG_PATH}).")
    parser.add_argument("--ss_db", default=cfg.SS_DB_PATH,
                        help=f"Path to secondary structure database (default: {cfg.SS_DB_PATH}).")
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed per-conformer logging.")
    args = parser.parse_args()
    main(args)
