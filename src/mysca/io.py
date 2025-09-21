"""Input/Output functions

"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from Bio import AlignIO
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.AlignIO import MultipleSeqAlignment

from mysca.mappings import SymMap, DEFAULT_MAP


def load_msa(
        fpath, 
        format: str = "fasta", 
        mapping: SymMap = DEFAULT_MAP,
        verbosity: int = 2,
) -> tuple[MultipleSeqAlignment, NDArray[np.int_], list]:
    """Load an MSA fasta file and return the MSA object, matrix, and IDs.
    
    Filters out any sequences that contain excluded characters in the given 
    mapping.

    Args:
        fpath (str): Path to input MSA file.
        format (str): Format of the input file. Default "fasta".
        mapping (SeqMap): SeqMap object defining the mapping from AAs to ints.
            Default is the default SeqMap defined in mappings.py.
        verbosity (int): verbosity level. Default 1.
    
    Returns:
        MultipleSeqAlignment: MSA object.
        NDArray[int]: Matrix representation of the MSA, as defined by the 
            given mapping.
        list[str]: Sequence IDs as defined in the input fasta file, that are 
            retained in the MSA.
    """
    msa_obj = AlignIO.read(fpath, format)

    # Keep records in the MSA not containing excluded symbols.
    exc_recs_screen = np.array([
        any([sym in str(record.seq) for sym in mapping.exclude_syms]) 
        for record in msa_obj
    ], dtype=bool)

    keep_records = [
        msa_obj[int(i)] for i in np.arange(len(msa_obj))[~exc_recs_screen]
    ]

    assert exc_recs_screen.sum() + len(keep_records) == len(msa_obj)

    msa_obj = MultipleSeqAlignment(keep_records)

    if verbosity > 1:
        print(f"Removed {exc_recs_screen.sum()} seqs with excluded syms.")

    # Construct the MSA matrix.
    msa_matrix = np.array([
        [mapping[aa] for aa in record.seq] for record in msa_obj 
        if np.all([excsym not in record.seq for excsym in mapping.exclude_syms])
    ])

    # Retrieve MSA sequence IDs.
    msa_ids = [record.id for record in msa_obj]
    return msa_obj, msa_matrix, msa_ids


def load_pdb_structure(fpath: str, id: str) -> Structure:
    """Load a PDB structure from a pdb file.
    
    Args:
        fpath (str): path to pdb file.
        id (str): the id for the returned structure.
    
    Returns:
        (Structure) Protein structure.
    """
    parser = PDBParser()
    structure = parser.get_structure(id, fpath)
    return structure

