"""
Centralized Configuration for the AlphaFlex-IDPForge Pipeline (Steps 1-4).

Author: SDC, 12/18/2025
"""

import os
import sys

# =============================================================================
# LOGGING CONTROL
# =============================================================================
# True  = Print every step (debugging / small batches)
# False = Progress bar only (HPC / large batches)
VERBOSE = True

# =============================================================================
# GLOBAL PATHS
# =============================================================================
PYTHON_EXEC = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Inputs")
MASTER_DB_PATH = os.path.join(INPUT_DATA_DIR, "Labeled_AlphaFlex_database_Nov2025.json")
LENGTH_REF_PATH = os.path.join(INPUT_DATA_DIR, "AF2_9606_HUMAN_v4_num_residues.json")
PDB_LIBRARY_PATH = os.path.join(INPUT_DATA_DIR, "AF2_9606_PAE_PDB")

PIPELINE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Pipeline_Outputs")

# =============================================================================
# STEP 1: CASE LABELING
# =============================================================================
STEP_1_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_1_Labeling")
LABELED_DB_PATH = os.path.join(STEP_1_DIR, "Labeled_AlphaFlex_database_Nov2025.json")
SUMMARY_TEXT_PATH = os.path.join(STEP_1_DIR, "idr_type_summary.txt")

# =============================================================================
# STEP 1B: SUBSET SAMPLING
# =============================================================================
ID_LISTS_OUTPUT_ROOT = os.path.join(STEP_1_DIR, "id_lists")
SUBSET_MIN_LENGTH = 0
SUBSET_MAX_LENGTH = 500
SUBSET_SAMPLE_SIZE = 100000

# =============================================================================
# STEP 2: TEMPLATE GENERATION
# =============================================================================
TEMPLATE_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_2_Templates")
IDP_CASES_LIST_PATH = os.path.join(TEMPLATE_OUTPUT_DIR, "idp_cases_to_run.json")

CURRENT_BATCH_FOLDER = "custom_subsets"
ID_LISTS_DIR = os.path.join(ID_LISTS_OUTPUT_ROOT, CURRENT_BATCH_FOLDER)

SCRIPT_STATIC_TEMPLATE = os.path.join(PARENT_DIR, "mk_ldr_template.py")
SCRIPT_FLEX_TEMPLATE = os.path.join(PARENT_DIR, "mk_flex_template.py")

TEMPLATE_N_CONFS = 200
TIMEOUT_STATIC_TEMPLATE = 60       # seconds
TIMEOUT_DYNAMIC_TEMPLATE = 1000    # seconds

# =============================================================================
# STEP 3: DIFFUSION SAMPLING
# =============================================================================
CONFORMER_POOL_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_3_Raw_Conformers")

SAMPLE_N_CONFS = 15
SAMPLE_BATCH_SIZE = 5
DEVICE = "cuda"

SCRIPT_SAMPLE_LDR = os.path.join(PARENT_DIR, "sample_ldr.py")
MODEL_WEIGHTS_PATH = os.path.join(PARENT_DIR, "weights", "mdl.ckpt")
MODEL_CONFIG_PATH = os.path.join(PARENT_DIR, "configs", "sample.yml")
SS_DB_PATH = os.path.join(PARENT_DIR, "data", "example_data.pkl")

# Adaptive clash scoring (Step 3)
SAMPLE_BASE_CLASH_THRESHOLD = 10.0
SAMPLE_CLASH_INCREMENT = 5.0
SAMPLE_MAX_TOTAL_ATTEMPTS = 500_000

# =============================================================================
# STEP 4: STITCHING & RELAXATION
# =============================================================================
STITCH_OUTPUT_ROOT = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_4_Final_Models")

# Stitching geometry
MIN_CONFORMER_POOL_SIZE = 100
STITCH_N_CONFORMERS = 100
ALIGNMENT_STUB_HALF_SIZE = 5
ALIGNMENT_JUNCTION_SIZE = 10

# AMBER relaxation
RELAX_STIFFNESS = 10.0
RELAX_MAX_OUTER_ITER = 20
MINIMIZATION_MAX_ITER = 0
MINIMIZATION_TOLERANCE = 10.0

# Adaptive clash scoring (Step 4)
STITCH_MAX_ATTEMPTS = 1_000_000
STITCH_BASE_CLASH_THRESHOLD = 10.0
STITCH_CLASH_INCREMENT = 5.0
