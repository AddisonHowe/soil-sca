#!/usr/bin/env bash

datdir=data/S1Ahalabi
seq_fname=S1Ahalabi_1470_nosnakes_noX.fasta
out_fname=MSA_S1Ahalabi_1470_nosnakes_noX

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
