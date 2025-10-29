#!/usr/bin/env bash

datdir=data/K00370
seq_fname=TIGR01580_noX_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57.fasta
out_fname=MSA_TIGR01580_noX_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57

nthreads=16

infile=${datdir}/seqs/${seq_fname}
outdir=${datdir}/msas
guidetreedir=${datdir}/guidetrees
outfile=${outdir}/${out_fname}.aln-fasta

echo "Aligning sequences in file ${infile}"
echo "Saving results to file ${outfile}"

mkdir -p $outdir
mkdir -p $guidetreedir

clustalo --infile ${infile} --outfile ${outfile} \
    --seqtype protein --outfmt fa --output-order tree-order --verbose \
    --guidetree-out ${guidetreedir}/${out_fname}.dnd  \
    --threads ${nthreads}
