#!/usr/bin/env bash

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PF00074_full
msafpath="data/PF00074/msas/PF00074_full.aln-fasta"
structdir="data/PF00074/structures"
outdir="out/PF00074/PF00074_full"
gap_truncation_thresh=1.0
sequence_gap_thresh=1.0
reference=None
reference_similarity_thresh=0.2
sequence_similarity_thresh=0.8
position_gap_thresh=1.0
regularization=0.03
background=None
n_top_conserved=3
n_boot=10
kstar=7

RUN_PYMOL=true
pymol_reference="3TGI"


# Run SCA script
mysca -msa $msafpath -s $structdir -o $outdir \
    --gap_truncation_thresh $gap_truncation_thresh \
    --sequence_gap_thresh $sequence_gap_thresh \
    --reference $reference \
    --reference_similarity_thresh $reference_similarity_thresh \
    --sequence_similarity_thresh $sequence_similarity_thresh \
    --position_gap_thresh $position_gap_thresh \
    --regularization $regularization \
    --background $background \
    --n_top_conserved $n_top_conserved \
    --n_boot $n_boot \
    --kstar $kstar \
    --pbar


# Run pymol script
if [[ ${RUN_PYMOL} -eq "true" ]]; then
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
            --groups 1 2 3
    done
fi
