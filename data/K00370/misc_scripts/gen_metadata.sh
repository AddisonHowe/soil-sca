#!/usr/bin/env bash

infile=data/K00370/seqs/TIGR01580.fasta
outfile=data/K00370/misc/metadata_TIGR01580.tsv

taxids_outfile=data/K00370/misc/taxids_TIGR01580.txt

awk '/^>/ {
    sub(/^>/, "");
    n = split($0, fields, "|");
    for (i = 1; i <= n; i++) {
        printf "%s%s", fields[i], (i < n ? "\t" : "\n")
    }
}' $infile > $outfile

cat $outfile | awk -F'\t' '{print substr($4,7)}' | sort -nu > $taxids_outfile
