#!/usr/bin/env bash

python scripts/partition_msa.py \
    -m data/K00370/msas/MSA_TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates.aln-fasta \
    -o data/K00370/msas \
    -n MSA_TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg \
    -p 750 1500
