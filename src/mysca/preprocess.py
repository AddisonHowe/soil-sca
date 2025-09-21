"""SCA Preprocessing

"""

import numpy as np
from numpy.typing import NDArray

from mysca.mappings import SymMap, DEFAULT_MAP


def preprocess_msa(
        msa: NDArray[np.int_], 
        seqids: list[str], 
        mapping: SymMap = DEFAULT_MAP,
        gap_truncation_thresh: float = 0.4,
        sequence_gap_thresh: float = 0.2, 
        reference_id: str = None,
        reference_similarity_thresh: float = 0.2,
        sequence_similarity_thresh: float = 0.8,
        position_gap_thresh: float = 0.2, 
        verbosity: int = 1, 
):
    """Run preprocessing steps on a given MSA matrix.

    Ref [1] Rivoire et al. 2016. https://doi.org/10.1371/journal.pcbi.1004817

    Args:
        (NDArray[np.int_]) msa: MSA object.
        (list[str]) seqids: IDs of sequences in the MSA.
        (SymMap) mapping: SymMap mapping symbols to integer values.
        (float) gap_truncation_thresh: Freq of gaps τ above which a position 
            (i.e. column) is removed for excessive gaps. Default 0.4.
        (float) sequence_gap_thresh: Freq of gaps γ_seq above which a sequence 
            (i.e. row) is removed. Default 0.2.
        (str) reference_id: ID of the reference sequence, or None.
        (float) reference_similarity_thresh: Identity threshold Δ below which
            sequences are excluded for not being close enough to the reference.
            Default 0.2.
        (float) sequence_similarity_thresh: Identity threshold δ above which 
            sequences are clustered together for weighting purposes. Default 0.8.
        (float) position_gap_thresh: Freq of gaps γ_pos above which a position 
            (i.e. column) is removed after weighting. Default 0.2.
        (int) verbosity: verbosity level. Default 1.
    
    Returns:
        (MultSeqAlignment) processed MSA.
        (NDArray[bool]) boolean MSA matrix after processing.
        (list[str]) retained sequence IDs.
        (NDArray[float]) sequence weights.
        (NDArray[int]) retained sequences.
        (NDArray[int]) retained positions.
        (dict): reference similarity results. If a reference ID is specified,
            will contain keys reference_id, ref_idx, and ref_similarity.
            
    """
    
    if verbosity:
        print("Preprocessing with parameters:")
        print(f"  gap_truncation_thresh τ={gap_truncation_thresh}")
        print(f"  sequence_gap_thresh γ_seq={sequence_gap_thresh}")
        print(f"  reference_id: {reference_id}")
        print(f"  reference_similarity_thresh Δ={reference_similarity_thresh}")
        print(f"  sequence_similarity_thresh δ={sequence_similarity_thresh}")
        print(f"  position_gap_thresh γ_pos={position_gap_thresh}")
    
    msa_orig = msa
    msa = msa_orig.copy()
    seqids_orig = seqids
    seqids = seqids_orig.copy()
    num_seqs, num_pos = msa_orig.shape

    if not isinstance(msa_orig, np.ndarray):
        raise RuntimeError(
            f"Input MSA should be an NDArray. Got {type(msa_orig)}"
        )
    if not isinstance(msa_orig[0,0], np.int_):
        raise RuntimeError(
            f"Input MSA should be an NDArray of ints. Got {type(msa_orig[0,0])}"
        )

    NUM_SYMS = len(mapping)
    GAP = mapping.gapint

    # Track which rows and columns will be kept
    retained_sequences = np.arange(num_seqs)
    retained_positions = np.arange(num_pos)

    # Constuct the boolean MSA matrix
    xmsa = np.eye(NUM_SYMS, dtype=bool)[msa][:,:,:-1]

    #~~~ Remove columns (i.e. positions) with too many gaps
    gapfreqs = np.sum(msa == GAP, axis=0) / msa.shape[0]
    screen = gapfreqs < gap_truncation_thresh
    msa = msa[:,screen]  # keep columns with gap freq < gap_truncation_thresh
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    if verbosity:
        print(f"Filtered {np.sum(~screen)} positions at threshold {gap_truncation_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_positions) == msa.shape[1], "Mismatch"

    #~~~ Remove rows (i.e. sequences) with too many gaps
    gapfreqs = np.sum(msa == GAP, axis=1) / msa.shape[1]
    screen = gapfreqs < sequence_gap_thresh
    msa = msa[screen,:]  # keep rows with gap freq < sequence_gap_thresh
    xmsa = xmsa[screen,:,:]
    retained_sequences = retained_sequences[screen]
    seqids = np.array([seqids_orig[i] for i in retained_sequences])
    if verbosity:
        print(f"Filtered {np.sum(~screen)} sequences at threshold {sequence_gap_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_sequences) == msa.shape[0], "Mismatch"

    #~~~ Compare with reference, if specified
    if reference_id:
        ref_idx = np.where(seqids == reference_id)[0][0]
        if verbosity:
            print(f"Found reference seq {reference_id} at position {ref_idx}.")
        refrow = msa[ref_idx,:]
        ref_similarity = np.sum(msa == refrow, axis = 1) / msa.shape[1]
        ref_results = {}
        ref_results["reference_id"] = reference_id
        ref_results["ref_idx"] = ref_idx
        ref_results["ref_similarity"] = ref_similarity
        
        # Remove rows too dissimilar from the reference
        screen = ref_similarity >= reference_similarity_thresh
        msa = msa[screen,:]  # keep rows with gap freq < reference_similarity_thresh
        xmsa = xmsa[screen,:,:]
        retained_sequences = retained_sequences[screen]
        seqids = np.array([seqids_orig[i] for i in retained_sequences])
        if verbosity:
            print(f"Filtered {np.sum(~screen)} sequences at threshold {reference_similarity_thresh}.")
            print(f"  MSA shape: {msa.shape} (sequences x positions)")
        assert len(retained_sequences) == msa.shape[0], "Mismatch"
    else:
        ref_results = {}

    #~~~ Compute sequence weights
    ws = np.nan * np.ones(msa.shape[0])
    for i, s in enumerate(msa):
        similarities = np.sum(s == msa, axis=1) / msa.shape[1]
        screen = similarities >= sequence_similarity_thresh
        ws[i] = 1 / screen.sum()

    #~~~ Remove positions with too many (weighted) gaps
    fi0 = np.sum(ws[:,None] * (msa == GAP), axis=0) / ws.sum()
    screen = fi0 < position_gap_thresh
    msa = msa[:,screen]
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    if verbosity:
        print(f"Filtered {np.sum(~screen)} positions at threshold {position_gap_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_positions) == msa.shape[1], "Mismatch"

    #~~~ Re-compute sequence weights
    ws = np.nan * np.ones(msa.shape[0])
    for i, s in enumerate(msa):
        similarities = np.sum(s == msa, axis=1) / msa.shape[1]
        screen = similarities >= sequence_similarity_thresh
        print(screen.sum())
        ws[i] = 1 / screen.sum()
    
    print(f"Effective sample size (sum of weights): {ws.sum()}")

    return msa, xmsa, seqids, ws, retained_sequences, retained_positions, ref_results
