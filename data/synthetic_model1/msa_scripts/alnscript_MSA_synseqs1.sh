#!/usr/bin/env bash

datdir=data/synthetic_model1
seq_fname=synseqs1.fasta
out_fname=synseqs1

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
