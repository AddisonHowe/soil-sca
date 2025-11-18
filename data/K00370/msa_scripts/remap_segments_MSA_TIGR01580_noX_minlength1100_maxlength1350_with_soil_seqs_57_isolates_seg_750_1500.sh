#!/usr/bin/env bash

python scripts/remap_positional_idxs.py \
    -d out/K00370/TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_0_0_750 \
    -m data/K00370/msas/MSA_TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_0_0_750.origpositions.npz

python scripts/remap_positional_idxs.py \
    -d out/K00370/TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_1_750_1500 \
    -m data/K00370/msas/MSA_TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_1_750_1500.origpositions.npz

python scripts/remap_positional_idxs.py \
    -d out/K00370/TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_2_1500_2279 \
    -m data/K00370/msas/MSA_TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_2_1500_2279.origpositions.npz
