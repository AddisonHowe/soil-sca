#!/usr/bin/env bash

RUN=(1 2 3 4 5 5b 5pymol 6 6b 6pymol 7)
# RUN=(5 5b 6 6b)
# RUN=(5all 6all)
# RUN=(5pymol 6pymol)


should_run() {
    local n=$1
    for i in "${RUN[@]}"; do
        [[ "$i" == "$n" ]] && return 0
    done
    return 1
}

datdirbase=out/K00370/TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57
outdirbase=out/figures/TIGR01580_noX_minlength1100_maxlength1350_with_soil_seqs_57
datdir=${datdirbase}/sca_results

seq_metadata_fpath=data/K00370/misc/metadata_TIGR01580_noX.tsv
taxa_metadata_fpath=data/K00370/misc/taxids_TIGR01580_metadata.tsv

groups_dir=${datdirbase}/sca_groups
scores_dir=${datdirbase}/pdb_sectors

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if should_run 1; then
    outdir=${outdirbase}/fig1
    python figure_scripts/gen_fig1.py \
        -d ${datdir} \
        -o ${outdir} \
        -v 1
fi

if should_run 2; then
    outdir=${outdirbase}/fig2
    python figure_scripts/gen_fig2.py \
        -d ${datdir} \
        -o ${outdir} \
        -v 1
fi

if should_run 3; then
    outdir=${outdirbase}/fig3
    python figure_scripts/gen_fig3.py \
        -d ${datdir} \
        -o ${outdir} \
        -v 1
fi

if should_run 4; then
    outdir=${outdirbase}/fig4
    python figure_scripts/gen_fig4.py \
        -d ${datdir} \
        -o ${outdir} \
        -v 1
fi

if should_run 5; then
    outdir=${outdirbase}/fig5
    python figure_scripts/gen_fig5.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --pairs 1 2  3 6 \
        -xl     0 0  0 0 \
        -yl     0 0  0 0 \
        -o ${outdir} \
        -v 1
fi

if should_run 5all; then
    outdir=${outdirbase}/fig5
    python figure_scripts/gen_fig5.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --pairs 0 1  1 2  3 4  5 6  7 8  9 10 \
        -xl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -yl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -o ${outdir} \
        -v 1
fi

if should_run 5b; then
    outdir=${outdirbase}/fig5b
    python figure_scripts/gen_fig5.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --pairs 3 6  \
        -xl   -0.03 -0.01  \
        -yl    0.01  0.03  \
        -o ${outdir} \
        -v 1
fi

if should_run 5pymol; then
    outdir=${outdirbase}/fig5
    structdir="data/K00370/structures"
    pymol_reference="1Q16"
    scaffolds=(
        "Soil3.scaffold_285743490_c1_2"
        "Soil3.scaffold_333288240_c1_8"
    )
    groups=(
        "0 1"
        "1 2"
        "3 6"
    )
    for scaffold in ${scaffolds[@]}; do
        struct_fpath=${structdir}/${f}.pdb
        for argval_groups in "${groups[@]}"; do
            for argval_multisector in "" "--multisector"; do
                python scripts/pymol_sca.py \
                    -s ${scaffold} \
                    -r ${pymol_reference} \
                    --pdb_dir ${structdir} \
                    --groups_dir ${groups_dir} \
                    --scores_dir ${scores_dir} \
                    --outdir ${outdir} \
                    --groups ${argval_groups} \
                    --show_molybdenum ${argval_multisector}
            done
        done
    done

    groups=(
        "0 1 2"
    )
    for scaffold in ${scaffolds[@]}; do
        struct_fpath=${structdir}/${f}.pdb
        for argval_groups in "${groups[@]}"; do
            python scripts/pymol_sca.py \
                -s ${scaffold} \
                -r ${pymol_reference} \
                --pdb_dir ${structdir} \
                --groups_dir ${groups_dir} \
                --scores_dir ${scores_dir} \
                --outdir ${outdir} \
                --groups ${argval_groups} \
                --show_molybdenum
        done
    done
fi

if should_run 6; then
    outdir=${outdirbase}/fig6
    python figure_scripts/gen_fig6.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --ranks phylum class \
        --pairs 0 1  1 2  3 4  5 6  7 8  3 6  \
        -xl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -yl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -o ${outdir} \
        -v 1
fi

if should_run 6all; then
    outdir=${outdirbase}/fig6
    python figure_scripts/gen_fig6.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --ranks phylum class \
        --pairs 0 1  1 2  3 4  5 6  7 8  9 10 \
        -xl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -yl     0 0  0 0  0 0  0 0  0 0  0 0  \
        -o ${outdir} \
        -v 1
fi


if should_run 6b; then
    outdir=${outdirbase}/fig6b
    python figure_scripts/gen_fig6.py \
        -d ${datdir} \
        -sm ${seq_metadata_fpath} \
        -tm ${taxa_metadata_fpath} \
        --ranks phylum class \
        --pairs 3 6  \
        -xl     -0.03 -0.01  \
        -yl     0.01 0.03  \
        -o ${outdir} \
        -v 1
fi

if should_run 6pymol; then
    outdir=${outdirbase}/fig6
    structdir="data/K00370/structures"
    pymol_reference="1Q16"
    scaffolds=(
        "Soil3.scaffold_285743490_c1_2"
    )
    groups=(
        "0 1"
        # "1 2"
        # "3 6"
    )
    for scaffold in ${scaffolds[@]}; do
        struct_fpath=${structdir}/${f}.pdb
        for argval_groups in "${groups[@]}"; do
            for argval_multisector in "" "--multisector"; do
                python scripts/pymol_sca.py \
                    -s ${scaffold} \
                    -r ${pymol_reference} \
                    --pdb_dir ${structdir} \
                    --groups_dir ${groups_dir} \
                    --scores_dir ${scores_dir} \
                    --outdir ${outdir} \
                    --groups ${argval_groups} \
                    --show_molybdenum ${argval_multisector}
            done
        done
    done
fi

if should_run 7; then
    outdir=${outdirbase}/fig7
    python figure_scripts/gen_fig7.py \
        -d ${datdir} \
        -o ${outdir} \
        -sd ${datdirbase}/groups \
        -si 1 2 3 6 \
        -v 1
fi
