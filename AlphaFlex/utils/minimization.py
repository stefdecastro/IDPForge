"""
utils/minimization.py
Pipeline orchestration for the relaxation and validation of conformers.
"""

import os
import io
import gc
from Bio.PDB import PDBIO
from openmm.app import PDBFile
from .pre_minimization import repair_chirality
from .post_minimization import validate_structure_post_relax

# Attempt to import the relaxation engine if available in the environment
try:
    from idpforge.utils.relax import relax_protein
except ImportError:
    relax_protein = None

def run_minimization(structure, output_filepath, idr_indices=None, device="cuda:0", attempts=0):
    """
    Full workflow for minimizing and validating a protein structure.
    """
    # 1. PRE-MINIMIZATION: Repair flipped chiral centers
    repair_chirality(structure)

    # Convert Bio.PDB to string buffer for OpenMM/OpenFold compatibility
    io_pdb = PDBIO()
    io_pdb.set_structure(structure)
    pdb_buf = io.StringIO()
    io_pdb.save(pdb_buf)
    
    pdb_name = os.path.splitext(os.path.basename(output_filepath))[0]
    output_dir = os.path.dirname(output_filepath)
    temp_relaxed_pdb = os.path.join(output_dir, f"{pdb_name}_relaxed.pdb")

    # 2. RELAXATION: Perform energy minimization
    if relax_protein is None:
        print("CRITICAL: relax_protein module not found.")
        return 0

    try:
        from openfold.np import protein as of_protein
        unrelaxed_prot = of_protein.from_pdb_string(pdb_buf.getvalue())
        # Run minimization (using config from your existing setup)
        success = relax_protein({}, device, unrelaxed_prot, output_dir, pdb_name)
        gc.collect()
    except Exception as e:
        print(f"       [CRASH] Relaxation failed: {e}")
        return 0

    # 3. POST-MINIMIZATION: Final quality gatekeeping
    if success == 1 and os.path.exists(temp_relaxed_pdb):
        chk = PDBFile(temp_relaxed_pdb)
        is_valid, _ = validate_structure_post_relax(
            topology=chk.topology,
            positions=chk.positions,
            pdb_path=temp_relaxed_pdb,
            idr_indices=idr_indices,
            attempts=attempts,
            verbose=True
        )
        
        if is_valid:
            if os.path.exists(output_filepath): os.remove(output_filepath)
            os.rename(temp_relaxed_pdb, output_filepath)
            return 1
        else:
            # Delete file if it fails post-min checks (e.g. chirality or knots)
            os.remove(temp_relaxed_pdb)
            
    return 0