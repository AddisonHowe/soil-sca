#!/usr/bin/env bash

python scripts/cutswap_sequences.py \
    -m data/K00370/msas/MSA_TIGR01580_with_soil_seqs_57.aln-fasta \
    -r results/TIGR01580_with_soil_seqs_57/sca_results \
    -o data/K00370/seqs \
    -n TIGR01580_cutswap_with_soil_seqs_57.fasta \
    --percentile 50
