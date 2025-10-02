#!/usr/bin/env bash

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ S1Ahalabi s1Ahalabi_1470_nosnakes
msafpath="data/S1Ahalabi/msas/s1Ahalabi_1470_nosnakes.aln-fasta"
structdir="data/S1Ahalabi/structures"
outdir="out/S1Ahalabi/s1Ahalabi_1470_nosnakes"
gap_truncation_thresh=0.4
sequence_gap_thresh=0.2
reference="gi|4139558|pdb|3TGI|E__vertebrate|warm|Rattus"
reference_similarity_thresh=0.2
sequence_similarity_thresh=0.8
position_gap_thresh=0.2
regularization=0.03
background=None
n_top_conserved=5
n_boot=0  # Note: set to 0
kstar=0

RUN_PYMOL=true
pymol_reference="3TGI"


# Run SCA script
runsca -msa $msafpath -o $outdir \
    --gap_truncation_thresh $gap_truncation_thresh \
    --sequence_gap_thresh $sequence_gap_thresh \
    --reference "$reference" \
    --reference_similarity_thresh $reference_similarity_thresh \
    --sequence_similarity_thresh $sequence_similarity_thresh \
    --position_gap_thresh $position_gap_thresh \
    --regularization $regularization \
    --background $background \
    --n_top_conserved $n_top_conserved \
    --n_boot $n_boot \
    --kstar $kstar \
    --pbar -v 10


# Run pymol script
if [[ ${RUN_PYMOL} == "true" ]]; then
    echo "Running pymol postscript..."
    for f in ${structdir}/*.pdb; do
        s=$(basename $f)
        s=${s/.pdb/}
        echo $s
        python scripts/pymol_sca.py \
            -s ${s} \
            -r ${pymol_reference} \
            --pdb_dir ${structdir} \
            --groups_dir ${outdir}/sca_groups \
            --outdir ${outdir}/pymol_images \
            --groups 1 2 3 4 5 6
    done
fi
