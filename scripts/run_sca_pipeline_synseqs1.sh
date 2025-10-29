#!/usr/bin/env bash

set -e

###~~~~~~~~~~ SYNSEQS1
msafpath="data/synthetic_model1/msas/synseqs1.aln-fasta"
# structdir=""
outdir="out/synthetic_model1/synseqs1"
gap_truncation_thresh=0.4
sequence_gap_thresh=0.2
reference=None
reference_similarity_thresh=0.2
sequence_similarity_thresh=0.8
position_gap_thresh=0.2
regularization=0.03
background=None
n_top_conserved=10
n_boot=10
kstar=0

RUN_PYMOL=false
pymol_reference=""
haltafter=0


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
    --pbar \
    --seed 15313 \
    --weak_assignment 0 \
    --save_all #--load_data ${outdir}/sca_results


# Run pymol script
# count=0
# if [[ ${RUN_PYMOL} == "true" ]]; then
#     echo "Running pymol postscript..."
#     for f in ${structdir}/*.pdb; do
#         s=$(basename $f)
#         s=${s/.pdb/}
#         if [[ "$s" == "$pymol_reference" ]]; then
#             continue
#         fi
#         python scripts/pymol_sca.py \
#             -s ${s} \
#             -r ${pymol_reference} \
#             --pdb_dir ${structdir} \
#             --groups_dir ${outdir}/sca_groups \
#             --scores_dir ${outdir}/pdb_sectors \
#             --outdir ${outdir}/pymol_images \
#             --groups -1 \
#             --show_molybdenum
#         ((count++))
#         if [[ $count -eq $haltafter ]]; then
#             break
#         fi
#     done
# fi
