"""
Modified from ESM2
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import typing as T
import torch
import numpy as np
import os
import re
import uuid

from esm.esmfold.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
)

from openfold.np.protein import Protein as OFProtein
from openfold.np.protein import to_pdb, from_pdb_string
from openfold.utils.feats import atom14_to_atom37

from idpforge.utils.definitions import (
    sstype_order, ss_lib, sstype_num, 
    coil_types, coil_sample_probs,
)
from idpforge.utils.np_utils import calc_rg, assign_rama
            


def onehot_to_ss(tokens, mask):
    return "".join([ss_lib[t] for t in tokens[mask]])
    
def encode_ss(
    ss: str,
    backbone_tor: T.Optional[torch.Tensor] = None,
    sample_coil: bool = False,
    chain_linker: T.Optional[str] = "C" * 25) -> torch.Tensor:

    if backbone_tor is not None:
        rama = assign_rama(torch.atan2(backbone_tor[..., 0], backbone_tor[..., 1]).numpy())
        ss = "".join([d if d in ["H", "E"] else r for d, r in zip(ss, rama)])
    elif sample_coil:
        ss = "".join([np.random.choice(list(coil_sample_probs.keys()), 
                                       p=list(coil_sample_probs.values())) if i in coil_types \
                      else i for i in ss])
    ss = re.sub(r'[AH]{6,}', lambda match: 'H' * len(match.group()), ss) 
    if chain_linker is None:
        chain_linker = ""
    chains = ss.split(":")
    seq = chain_linker.join(chains)
    return torch.tensor([sstype_order.get(s) for s in seq])
    
def batch_encode_ss(
    ss: T.Sequence[str],
    backbone_tor: T.Optional[T.Sequence[torch.Tensor]] = None,
    chain_linker: T.Optional[str] = "C" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor]:

    sstype_list = []
    if backbone_tor is None:
        backbone_tor = [None] * len(ss) 
    for s, tor in zip(ss, backbone_tor):
        sstype_seq = encode_ss(
            s, tor, # (N, 2, 2)
            s.count("B")+s.count("A")==0,
            chain_linker=chain_linker,
        )
        sstype_list.append(sstype_seq)

    sstype = collate_dense_tensors(sstype_list)
    mask = collate_dense_tensors(
        [sstype.new_ones(len(sstype_seq)) for sstype_seq in sstype_list]
    )

    return sstype, mask

def input_process(
    sequences: T.Sequence[str], 
    ss: T.Sequence[str],
    backbone_tor: T.Optional[T.Sequence[torch.Tensor]] = None,
    residx = None,
    residue_index_offset: T.Optional[int] = 512, 
    chain_linker: T.Optional[str] = "G" * 25):
    aatype, aa_mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )
    sstype, ss_mask = batch_encode_ss(ss, backbone_tor, chain_linker.replace("G", "C"))
    assert torch.equal(aa_mask, ss_mask)

    if residx is None:
        residx = _residx
    elif not isinstance(residx, torch.Tensor):
        residx = collate_dense_tensors(residx)
    return aatype, sstype, aa_mask, residx, linker_mask
    

def output_to_pdb(
    output: dict,
    select_idx=None,
    relax=None,
    save_path=None,
    counter=1,
    counter_cap=None,
    **kwargs
):
    """
    Writes ATOM37 PDBs (with hydrogens) using numbered filenames.
    Compatible with OpenMM relaxation and post-min validation.

    Args:
        counter: Starting file index (1-based). Files are named {counter}_raw.pdb.
        counter_cap: If set, stops writing once counter exceeds this value.
    """

    # Convert atom14 → atom37 (IDPForge uses atom14 internally)
    final_atom_positions = atom14_to_atom37(output["positions"], output)
    final_atom_positions = final_atom_positions.numpy()

    # Convert all tensors to numpy
    output = {k: v.numpy() for k, v in output.items()}
    final_atom_mask = output["atom37_atom_exists"].astype(bool)

    if select_idx is None:
        select_idx = range(output["aatype"].shape[0])

    written_files = []
    file_idx = counter

    for i in select_idx:
        if counter_cap is not None and file_idx > counter_cap:
            break

        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1  # preserve numbering

        # Build full atom37 protein object
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=np.zeros(pred_pos.shape[:2]),
            chain_index=np.zeros(pred_pos.shape[0], dtype=int),
        )

        pdb_str = to_pdb(pred)

        # Numbered filename
        if save_path is not None:
            fname = os.path.join(save_path, f"{file_idx}_raw.pdb")
            with open(fname, "w") as f:
                f.write(pdb_str)
            written_files.append(fname)
        else:
            written_files.append(pdb_str)

    # If relaxation is requested, run it here
    if relax is not None:
        from idpforge.utils.relax import relax_protein
        relaxed_files = []
        for raw_path in written_files:
            stem = os.path.basename(raw_path).replace("_raw.pdb", "")
            relax_protein(
                relax,
                "cuda" if torch.cuda.is_available() else "cpu",
                from_pdb_string(open(raw_path).read()),
                save_path,
                stem,
                **kwargs
            )
            relaxed_files.append(os.path.join(save_path, stem + "_relaxed.pdb"))
        return relaxed_files

    return written_files

    

