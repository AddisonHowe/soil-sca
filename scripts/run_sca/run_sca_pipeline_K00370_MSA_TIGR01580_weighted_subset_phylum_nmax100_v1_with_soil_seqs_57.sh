#!/usr/bin/env bash

set -e

RUN_PYMOL=true
N_BOOT=0
LOAD_DIR=false
USE_JAX=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runpymol) RUN_PYMOL=true; shift ;;
        --no-runpymol) RUN_PYMOL=false; shift ;;
        --jax) USE_JAX=true; shift ;;
        --no-jax) USE_JAX=false; shift ;;
        -n|--n_boot) N_BOOT="$2"; shift 2 ;;
        --load) LOAD_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--runpymol|--no-runpymol]"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done


###~~~~~~~~~~ K00370 MSA_TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57
msafpath="data/K00370/msas/MSA_TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57.aln-fasta"
structdir="data/K00370/structures"
outdir="out/K00370/TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57"
gap_truncation_thresh=0.4
sequence_gap_thresh=0.2
reference=Soil11.scaffold_431547323_c1_2
reference_similarity_thresh=0.2
sequence_similarity_thresh=0.8
position_gap_thresh=0.2
regularization=0.03
background=None
n_top_conserved=10
n_boot=${N_BOOT}
kstar=0
pstar=95

if [[ "${LOAD_DIR}" == "false" ]]; then
    PYLOAD_ARG=""
elif [[ "${LOAD_DIR}" == "existing" ]]; then
    PYLOAD_ARG="--load_data ${outdir}/sca_results"
else
    PYLOAD_ARG="--load_data ${LOAD_DIR}"
fi

if [[ "${USE_JAX}" == "false" ]]; then
    JAX_ARG=""
else
    JAX_ARG="--use_jax"
fi

run_pymol=${RUN_PYMOL}
pymol_reference="1Q16"
haltafter=5


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
    --pstar $pstar \
    --pbar \
    --seed 6125 \
    --weak_assignment 0 \
    --save_all \
    ${PYLOAD_ARG} ${JAX_ARG}


# Run pymol script
count=0
if [[ ${run_pymol} == "true" ]]; then
    echo "Running pymol postscript..."
    for f in ${structdir}/*.pdb; do
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
            --scores_dir ${outdir}/pdb_sectors \
            --outdir ${outdir}/pymol_images \
            --groups -1 \
            --show_molybdenum --animate
        ((count++))
        if [[ $count -eq $haltafter ]]; then
            break
        fi
    done
fi
