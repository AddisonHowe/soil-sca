"""Preprocessing tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np

from mysca.io import load_msa
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa


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

        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "fa_fpath, symmap, " \
        "gap_truncation_thresh, sequence_gap_thresh, " \
        "reference_id, reference_similarity_thresh, " \
        "sequence_similarity_thresh, position_gap_thresh, " \
        "retained_sequences_exp, retained_positions_exp, weights_exp", [
    # Test MSA: msa01.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA1, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA1, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(5),
        np.arange(10),
        None
    ],

    # Test MSA: msa02.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA2, SYMMAP1_EXC_X,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA2, SYMMAP1_EXC_X,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.arange(2),
        np.arange(10),
        None
    ],

    # Test MSA: msa03.faa
    [# Keep all positions and sequences, regardless of gaps
        TEST_MSA3, SYMMAP1,
        1.0, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        None
    ],
    [# Keep positions with fewer than 50% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.5, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 4, 5, 6, 7, 8, 9]),  # remove position 3
        None
    ],
    [# Keep positions with fewer than 40% gaps. Keep all sequences.
        TEST_MSA3, SYMMAP1,
        0.4, 1.0,
        None, 1.0, 
        1.0, 1.0, 
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 50% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.5,
        None, 1.0, 
        1.0, 1.0, 
        np.array([1, 2, 3, 4]),  # remove sequence 0
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None
    ],
    [# Keep positions with fewer than 40% gaps. Keep sequences with fewer than 20% gaps.
        TEST_MSA3, SYMMAP1,
        0.4, 0.2,
        None, 1.0, 
        1.0, 1.0, 
        np.array([3, 4]),  # remove sequence 0, 1, 2
        np.array([0, 2, 6, 8]),  # remove position 1, 3, 4, 5, 7, 9
        None
    ],

    # Test MSA: msa04.faa
    [
        TEST_MSA4, SYMMAP2,
        0.4, 0.2, 
        "msa04_sequence1", 0.499, 
        1.0, 0.2, 
        np.arange(20),  # remove sequences 20, 21, 22
        np.concatenate(  # remove positions 10, 16, 17
            [np.arange(10), np.arange(11, 16), np.arange(18, 22)]
        ), 
        np.array([
            0.25, 0.1, 0.25, 0.25, 0.25, 0.1, 0.2, 0.2, 0.1, 
            0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1
        ])
    ],

])
def test_preprocessing_excessive_gaps(
    fa_fpath, symmap, 
    gap_truncation_thresh,
    sequence_gap_thresh,
    sequence_similarity_thresh,
    reference_id,
    reference_similarity_thresh, 
    position_gap_thresh,
    retained_sequences_exp,
    retained_positions_exp,
    weights_exp,
):
    
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

    msa, xmsa, seqids, weights, retained_sequences, retained_positions, _ = results

    errors = []
    if len(retained_sequences_exp) != len(retained_sequences) or \
            np.any(retained_sequences_exp != retained_sequences):
        msg = "Mismatch in retained sequences. "
        msg += f"Expected {retained_sequences_exp}. Got {retained_sequences}"
        errors.append(msg)
    if len(retained_positions_exp) != len(retained_positions) or \
            np.any(retained_positions_exp != retained_positions):
        msg = "Mismatch in retained positions. "
        msg += f"Expected {retained_positions_exp}. Got {retained_positions}"
        errors.append(msg)
    if weights_exp is not None:
        if not np.allclose(weights_exp, weights):
            msg = "Mismatch in weights. "
            msg += f"Expected {weights_exp}.\nGot {weights}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
