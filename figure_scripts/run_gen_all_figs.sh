#!/usr/bin/env bash

datdirbase=out/K00370/TIGR01580_with_soil_seqs_1000
outdirbase=out/figures/TIGR01580_with_soil_seqs_1000

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig1
# python figure_scripts/gen_fig1.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig2
# python figure_scripts/gen_fig2.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig3
# python figure_scripts/gen_fig3.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig4
# python figure_scripts/gen_fig4.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

datdir=${datdirbase}/sca_results
outdir=${outdirbase}/fig5
python figure_scripts/gen_fig5.py \
    -d ${datdir} \
    -sm data/K00370/misc/metadata_TIGR01580.tsv \
    -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
    --pairs 1 2 3 5 \
    -xl 0 0 -0.03 -0.01 \
    -yl 0 0 -0.03 -0.01 \
    -o ${outdir} \
    -v 1
# structdir="data/K00370/structures"
# pymol_reference="1Q16"
# scaffolds=(
#     "Soil3.scaffold_285743490_c1_2"
#     "Soil3.scaffold_333288240_c1_8"
# )
# for scaffold in ${scaffolds[@]}; do
#     struct_fpath=${structdir}/${f}.pdb
#     python scripts/pymol_sca.py \
#         -s ${scaffold} \
#         -r ${pymol_reference} \
#         --pdb_dir ${structdir} \
#         --groups_dir ${datdirbase}/sca_groups \
#         --outdir ${outdir} \
#         --groups 1 2 \
#         --multisector_group_idxs 1 2
#     python scripts/pymol_sca.py \
#         -s ${scaffold} \
#         -r ${pymol_reference} \
#         --pdb_dir ${structdir} \
#         --groups_dir ${datdirbase}/sca_groups \
#         --outdir ${outdir} \
#         --groups 3 5 \
#         --multisector_group_idxs 3 5
# done

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig6
# python figure_scripts/gen_fig6.py \
#     -d ${datdir} \
#     -sm data/K00370/misc/metadata_TIGR01580.tsv \
#     -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig7
# python figure_scripts/gen_fig7.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -sd ${datdirbase}/groups \
#     -si 1 2 3 5 \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/figN
# python figure_scripts/gen_figN.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1


##############################################################################
##############################################################################

datdirbase=out/K00370/TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_1000
outdirbase=out/figures/TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_1000

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig1
# python figure_scripts/gen_fig1.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig2
# python figure_scripts/gen_fig2.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig3
# python figure_scripts/gen_fig3.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig4
# python figure_scripts/gen_fig4.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1

datdir=${datdirbase}/sca_results
outdir=${outdirbase}/fig5
python figure_scripts/gen_fig5.py \
    -d ${datdir} \
    -sm data/K00370/misc/metadata_TIGR01580.tsv \
    -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
    --pairs 0 1 2 4 \
    -xl 0 0 -0.03 -0.01 \
    -yl 0 0 -0.03 -0.01 \
    -o ${outdir} \
    -v 1
# structdir="data/K00370/structures"
# pymol_reference="1Q16"
# scaffolds=(
#     "Soil3.scaffold_285743490_c1_2"
#     "Soil3.scaffold_333288240_c1_8"
# )
# for scaffold in ${scaffolds[@]}; do
#     struct_fpath=${structdir}/${f}.pdb
#     python scripts/pymol_sca.py \
#         -s ${scaffold} \
#         -r ${pymol_reference} \
#         --pdb_dir ${structdir} \
#         --groups_dir ${datdirbase}/sca_groups \
#         --outdir ${outdir} \
#         --groups 0 1 \
#         --multisector_group_idxs 0 1
#     python scripts/pymol_sca.py \
#         -s ${scaffold} \
#         -r ${pymol_reference} \
#         --pdb_dir ${structdir} \
#         --groups_dir ${datdirbase}/sca_groups \
#         --outdir ${outdir} \
#         --groups 2 4 \
#         --multisector_group_idxs 2 4
# done

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig6
# python figure_scripts/gen_fig6.py \
#     -d ${datdir} \
#     -sm data/K00370/misc/metadata_TIGR01580.tsv \
#     -tm data/K00370/misc/taxids_TIGR01580_metadata.tsv \
#     -o ${outdir} \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/fig7
# python figure_scripts/gen_fig7.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -sd ${datdirbase}/groups \
#     -si 0 1 2 4 \
#     -v 1

# datdir=${datdirbase}/sca_results
# outdir=${outdirbase}/figN
# python figure_scripts/gen_figN.py \
#     -d ${datdir} \
#     -o ${outdir} \
#     -v 1