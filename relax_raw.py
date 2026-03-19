#!/usr/bin/env python3
"""
Relax existing raw PDB files using the same Amber relaxation as sample_ldr.py.

Usage:
    python relax_raw.py ./A2RUH7_raw path/to/template.npz configs/sample.yml --cuda
    python relax_raw.py ./A2RUH7_raw path/to/template.npz configs/sample.yml --cuda --output_dir ./A2RUH7_relaxed
"""

import os
import sys
import argparse
import yaml
import re
import numpy as np
import torch
import ml_collections as mlc

from glob import glob
from openfold.np.protein import from_pdb_string
from idpforge.utils.relax import relax_protein

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="Relax existing *_raw.pdb files with Amber minimization.")
    parser.add_argument("raw_dir", help="Directory containing *_raw.pdb files")
    parser.add_argument("template_npz", help="Template .npz file (for mask → exclude_residues)")
    parser.add_argument("sample_cfg", help="Config YAML (e.g. configs/sample.yml)")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for relaxation")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as raw_dir)")
    args = parser.parse_args()

    # 1. Load template mask → exclude_residues
    fold_data = np.load(args.template_npz, allow_pickle=True)
    mask = fold_data["mask"]  # True = folded, False = IDR
    exclude_residues = np.where(~mask)[0].tolist() # ~mask = IDR residues, which are free to move during relaxation
    print(f"[relax] Loaded template: {args.template_npz}")
    print(f"[relax] exclude_residues: {len(exclude_residues)} IDR residues (free to move)")

    # 2. Load relax config from YAML
    with open(args.sample_cfg, "r") as f:
        settings = yaml.safe_load(f)

    relax_config = settings["relax"]
    relax_config["exclude_residues"] = exclude_residues
    relax_opts = mlc.ConfigDict(relax_config)
    print(f"[relax] Relax config: stiffness={relax_config.get('stiffness')}, "
          f"max_outer_iterations={relax_config.get('max_outer_iterations')}")

    # 3. Find raw PDBs, sorted numerically
    raw_files = sorted(
        glob(os.path.join(args.raw_dir, "*_raw.pdb")),
        key=lambda f: int(re.search(r"/(\d+)_raw\.pdb$", f).group(1))
    )
    if not raw_files:
        print(f"[relax] No *_raw.pdb files found in {args.raw_dir}")
        sys.exit(1)

    print(f"[relax] Found {len(raw_files)} raw PDB files")

    # 4. Setup output
    output_dir = args.output_dir or args.raw_dir
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 5. Relax each raw PDB
    success = 0
    failed = 0
    for raw_path in raw_files:
        basename = os.path.basename(raw_path)
        stem = basename.replace("_raw.pdb", "")
        print(f"\n[relax] Processing {basename}...")

        with open(raw_path, "r") as f:
            pdb_str = f.read()

        unrelaxed_protein = from_pdb_string(pdb_str)

        # viol_mask: same as sample_ldr.py line 154
        viol_mask = ~mask

        result = relax_protein(
            relax_opts,
            device,
            unrelaxed_protein,
            output_dir,
            stem,
            viol_mask=viol_mask,
        )

        if result == 1:
            print(f"[relax]   → {stem}_relaxed.pdb written")
            success += 1
        else:
            print(f"[relax]   → FAILED (violation threshold or minimization error)")
            failed += 1

    print(f"\n[relax] Done. {success} relaxed, {failed} failed out of {len(raw_files)} total.")


if __name__ == "__main__":
    main()
