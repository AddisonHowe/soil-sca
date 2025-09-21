"""Core tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import load_msa
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa
from mysca.core import run_sca


#####################
##  Configuration  ##
#####################

SYMMAP1 = SymMap("ACDEF", '-')
SYMMAP1_EXC_X = SymMap("ACDEF", '-', "X")

SYMMAP2 = SymMap("ABCDEFGH", '-')

TEST_MSA1 = f"{DATDIR}/msas/msa01.faa"
TEST_MSA2 = f"{DATDIR}/msas/msa02.faa"
TEST_MSA3 = f"{DATDIR}/msas/msa03.faa"
TEST_MSA4 = f"{DATDIR}/msas/msa04.faa"
TEST_MSA5 = f"{DATDIR}/msas/msa05.faa"

BACKGROUND_MAP1 = {}

LAMBDA1 = 0.03

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "fa_fpath, symmap, " \
        "gap_truncation_thresh, sequence_gap_thresh, " \
        "reference_id, reference_similarity_thresh, " \
        "sequence_similarity_thresh, position_gap_thresh, " \
        "background_map, regularization", [
    # Test MSA: msa01.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA1, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],

    # Test MSA: msa02.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA2, SYMMAP1_EXC_X,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],

    # Test MSA: msa03.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA3, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 50% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.5,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 20% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.2,
        None, 1.0, 
        1.0, 1.0, 
        None, LAMBDA1,
    ],

    # Test MSA: msa04.faa
    [
        TEST_MSA4, SYMMAP2,
        0.4, 0.2, 
        "msa04_sequence1", 0.499, 
        1.0, 0.2,
        None, LAMBDA1,
    ],

    # Test MSA: msa05.faa
    [
        TEST_MSA5, SYMMAP2,
        0.4, 0.2, 
        "msa05_sequence1", 0.499, 
        1.0, 0.2,
        None, LAMBDA1,
    ],

])
def test_run_sca(
    fa_fpath, symmap, 
    gap_truncation_thresh,
    sequence_gap_thresh,
    sequence_similarity_thresh,
    reference_id,
    reference_similarity_thresh, 
    position_gap_thresh,
    background_map,
    regularization,
):
    
    # Equal background probability distribution if background_map is None
    if background_map is None:
        background_map = {s: 1 / len(symmap.aa2int) for s in symmap.aa2int}
    
    msa_obj, msa_orig, msa_ids_orig = load_msa(
        fa_fpath, format="fasta", mapping=symmap,
    )

    results = preprocess_msa(
        msa_orig, msa_ids_orig, 
        mapping=symmap, 
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        verbosity=2
    )

    msa, xmsa, seqids, weights, ret_seqs, ret_pos, _ = results

    sca_res = run_sca(
        xmsa, weights, background_map,
        mapping=symmap,
        regularization=regularization,
        return_keys="all",
        pbar=False,
    )

    errors = []
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
