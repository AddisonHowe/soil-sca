#!/usr/bin/env bash

datdirbase=out/K00370/TIGR01580_with_soil_seqs_57
outdirbase=out/figures/TIGR01580_with_soil_seqs_57
datdir=${datdirbase}/sca_results

outdir=${outdirbase}/fig1
python figure_scripts/gen_fig1.py \
    -d ${datdir} \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig2
python figure_scripts/gen_fig2.py \
    -d ${datdir} \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig3
python figure_scripts/gen_fig3.py \
    -d ${datdir} \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig4
python figure_scripts/gen_fig4.py \
    -d ${datdir} \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig5
python figure_scripts/gen_fig5.py \
    -d ${datdir} \
    -sm data/K00370/misc/metadata_TIGR01580.tsv \
    -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
    --pairs 1 2 3 5 \
    -xl 0 0  0 0 \
    -yl 0 0  0 0 \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig5
structdir="data/K00370/structures"
pymol_reference="1Q16"
scaffolds=(
    "Soil3.scaffold_285743490_c1_2"
    "Soil3.scaffold_333288240_c1_8"
)
groups=(
    "1 2"
    "3 5"
)
for scaffold in ${scaffolds[@]}; do
    struct_fpath=${structdir}/${f}.pdb
    for argval_groups in "${groups[@]}"; do
        for argval_multisector in "" "--multisector"; do
            python scripts/pymol_sca.py \
                -s ${scaffold} \
                -r ${pymol_reference} \
                --pdb_dir ${structdir} \
                --groups_dir ${datdirbase}/sca_groups \
                --scores_dir ${datdirbase}/pdb_sectors \
                --outdir ${outdir} \
                --groups ${argval_groups} \
                --show_molybdenum ${argval_multisector}
        done
    done
done

outdir=${outdirbase}/fig6
python figure_scripts/gen_fig6.py \
    -d ${datdir} \
    -sm data/K00370/misc/metadata_TIGR01580.tsv \
    -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
    --ranks phylum \
    --pairs 0 1  1 2  3 4  5 6  7 8  3 5  \
    -xl     0 0  0 0  0 0  0 0  0 0  0 0  \
    -yl     0 0  0 0  0 0  0 0  0 0  0 0  \
    -o ${outdir} \
    -v 1

outdir=${outdirbase}/fig6
structdir="data/K00370/structures"
pymol_reference="1Q16"
scaffolds=(
    "Soil3.scaffold_285743490_c1_2"
)
groups=(
    "1 2"
    "3 5"
)
for scaffold in ${scaffolds[@]}; do
    struct_fpath=${structdir}/${f}.pdb
    for argval_groups in "${groups[@]}"; do
        for argval_multisector in "" "--multisector"; do
            python scripts/pymol_sca.py \
                -s ${scaffold} \
                -r ${pymol_reference} \
                --pdb_dir ${structdir} \
                --groups_dir ${datdirbase}/sca_groups \
                --scores_dir ${datdirbase}/pdb_sectors \
                --outdir ${outdir} \
                --groups ${argval_groups} \
                --show_molybdenum ${argval_multisector}
        done
    done
done

outdir=${outdirbase}/fig7
python figure_scripts/gen_fig7.py \
    -d ${datdir} \
    -o ${outdir} \
    -sd ${datdirbase}/groups \
    -si 1 2 3 5 \
    -v 1

# outdir=${outdirbase}/figN
# python figure_scripts/gen_figN.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1
