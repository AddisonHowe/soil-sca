#!/usr/bin/env bash

rank=phylum
nmax=100

# Construct a reproducible seed addition from the rank string
seed=$(echo -n "${rank}" | cksum | awk '{print $1}')

python data/K00370/misc_scripts/subset_TIGR01580_sequences.py \
    -i data/K00370/seqs/TIGR01580.fasta \
    -n ${nmax} \
    -o data/K00370/seqs/TIGR01580_weighted_subset_${rank}_nmax${nmax}_v1.fasta \
    -sm data/K00370/misc/metadata_TIGR01580.tsv \
    -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
    -ro data/K00370/misc/rankcounts_TIGR01580 \
    -r $rank --seed $((42 + seed))
