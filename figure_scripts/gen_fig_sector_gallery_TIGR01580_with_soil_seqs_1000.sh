#!/usr/bin/env bash

datdirbase=out/K00370/TIGR01580_with_soil_seqs_1000
outdirbase=out/figures/TIGR01580_with_soil_seqs_1000
datdir=${datdirbase}/sca_results

outdir=${outdirbase}/fig_sector_gallery

structdir="data/K00370/structures"
pymol_reference="1Q16"
scaffolds=(
    "Soil3.scaffold_285743490_c1_2"
)

groups=(
    "-1"
)

for scaffold in ${scaffolds[@]}; do
    struct_fpath=${structdir}/${f}.pdb
    for argval_groups in "${groups[@]}"; do
        python scripts/pymol_sca.py \
            -s ${scaffold} \
            -r ${pymol_reference} \
            --pdb_dir ${structdir} \
            --groups_dir ${datdirbase}/sca_groups \
            --scores_dir ${datdirbase}/pdb_sectors \
            --outdir ${outdir} \
            --groups ${argval_groups} \
            --show_molybdenum --animate
    done
done
