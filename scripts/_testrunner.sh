#!/usr/bin/env bash

set -e

###~~~~~~~~~~ K00370 MSA_1000
msafpath="data/K00370/msas/MSA_1000.aln-fasta"
structdir="data/K00370/structures"
outdir="out/K00370/MSA_1000"
gap_truncation_thresh=0.4
sequence_gap_thresh=0.2
reference=None
reference_similarity_thresh=0.2
sequence_similarity_thresh=0.8
position_gap_thresh=0.2
regularization=0.03
background=None
n_top_conserved=10
n_boot=0
kstar=0

RUN_PYMOL=true
pymol_reference="1Q16"


# Run SCA script
# runsca -msa $msafpath -o $outdir \
#     --gap_truncation_thresh $gap_truncation_thresh \
#     --sequence_gap_thresh $sequence_gap_thresh \
#     --reference "$reference" \
#     --reference_similarity_thresh $reference_similarity_thresh \
#     --sequence_similarity_thresh $sequence_similarity_thresh \
#     --position_gap_thresh $position_gap_thresh \
#     --regularization $regularization \
#     --background $background \
#     --n_top_conserved $n_top_conserved \
#     --n_boot $n_boot \
#     --kstar $kstar \
#     --pbar \
#     --nodendro --save_all --load_data ${outdir}/sca_results


# Run pymol script
if [[ ${RUN_PYMOL} == "true" ]]; then
    echo "Running pymol postscript..."
    count=0
    for f in ${structdir}/*.pdb; do
        if [[ $count -eq 2 ]]; then
            exit 0
        fi
        s=$(basename $f)
        s=${s/.pdb/}
        if [[ "$s" == "$pymol_reference" ]]; then
            continue
        fi
        python scripts/pymol_sca.py \
            -s ${s} \
            -r ${pymol_reference} \
            --pdb_dir ${structdir} \
            --groups_dir ${outdir}/sca_groups \
            --outdir ${outdir}/delme_pymol_images \
            --groups -1
        ((count++))
    done
fi
