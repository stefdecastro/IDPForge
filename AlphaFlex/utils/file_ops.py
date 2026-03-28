import os
import shutil
import re

# ------------------------------------------------------------
# 1. Transfer Files
# ------------------------------------------------------------
def atomic_transfer(src, dest_dir, dest_name):
    temp_dest = os.path.join(dest_dir, f".tmp_{dest_name}")
    final_dest = os.path.join(dest_dir, dest_name)
    try:
        shutil.copy2(src, temp_dest)
        os.replace(temp_dest, final_dest)
        if os.path.exists(src):
            os.remove(src)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# 2. Clean final directory and fix misnamed files
# ------------------------------------------------------------
def rename_and_clean_final_directory(final_dir, log_func=None):
    """
    Ensures the final directory contains only properly named files:
        attN_relaxed.pdb  OR  N_relaxed.pdb

    - Removes phantom files
    - Removes stray temp files
    - Renames misnumbered files to a consistent sequence
    - Returns the next available attempt index
    """

    if not os.path.exists(final_dir):
        return 0

    files = os.listdir(final_dir)
    valid = []
    removed = 0

    for f in files:
        path = os.path.join(final_dir, f)

        # --- NEW: skip directories (e.g., _raw_staging) ---
        if os.path.isdir(path):
            continue

        # Remove phantom files
        if f.startswith(".tmp_"):
            os.remove(path)
            removed += 1
            continue

        # Remove staging leftovers
        if f.endswith(".tmp") or f.endswith(".bak"):
            os.remove(path)
            removed += 1
            continue

        # Accept only *_relaxed.pdb
        if re.match(r"^\d+_relaxed\.pdb$", f) or re.match(r"^att\d+_relaxed\.pdb$", f):
            valid.append(f)
        else:
            # Unknown file → remove
            os.remove(path)
            removed += 1

    # Sort numerically by the number before "_relaxed"
    def extract_num(name):
        m = re.match(r"^(?:att)?(\d+)_relaxed\.pdb$", name)
        return int(m.group(1)) if m else 999999

    valid_sorted = sorted(valid, key=extract_num)

    # Renumber them cleanly: 0_relaxed.pdb, 1_relaxed.pdb, ...
    for i, old_name in enumerate(valid_sorted):
        old_path = os.path.join(final_dir, old_name)
        new_name = f"{i}_relaxed.pdb"
        new_path = os.path.join(final_dir, new_name)
        if old_path != new_path:
            os.replace(old_path, new_path)

    if log_func:
        log_func(f"   [CLEAN] Removed {removed} stray files. Kept {len(valid_sorted)} valid conformers.")

    return len(valid_sorted)

# ------------------------------------------------------------
# 3. Cleanup staging area
# ------------------------------------------------------------
def cleanup_staging_area(staging_dir, force=False):
    """
    Removes leftover raw files and phantom files from the staging directory.
    If force=True, removes the entire directory.
    """

    if not os.path.exists(staging_dir):
        return

    if force:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    for f in os.listdir(staging_dir):
        path = os.path.join(staging_dir, f)

        # Remove raw attempts
        if f.startswith("raw_att") and f.endswith(".pdb"):
            os.remove(path)
            continue

        # Remove phantom files
        if f.startswith(".tmp_") or f.endswith(".tmp"):
            os.remove(path)
            continue

        # Remove anything unexpected
        if not f.endswith(".pdb"):
            os.remove(path)


# ------------------------------------------------------------
# 4. Optional renumbering utility (already in your file)
# ------------------------------------------------------------
def sanitize_and_renumber(directory, prefix="model_", suffix=".pdb", target_count=100):
    files = sorted([f for f in os.listdir(directory) if f.endswith(suffix)])
    for i, f in enumerate(files):
        os.rename(os.path.join(directory, f), os.path.join(directory, f"{prefix}{i}{suffix}"))
    return len(files)