#!/usr/bin/env bash

set -e

RUN_PYMOL=true
N_BOOT=0
LOAD_DIR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runpymol) RUN_PYMOL=true; shift ;;
        --no-runpymol) RUN_PYMOL=false; shift ;;
        -n|--n_boot) N_BOOT="$2"; shift 2 ;;
        --load) LOAD_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--runpymol|--no-runpymol]"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done


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
n_boot=20
kstar=0
pstar=95

if [[ "${LOAD_DIR}" == "false" ]]; then
    PYLOAD_ARG=""
elif [[ "${LOAD_DIR}" == "existing" ]]; then
    PYLOAD_ARG="--load_data ${outdir}/sca_results"
else
    PYLOAD_ARG="--load_data ${LOAD_DIR}"
fi

RUN_PYMOL=${RUN_PYMOL}
pymol_reference="3TGI"
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
    --seed 578347 \
    --save_all \
    ${PYLOAD_ARG}


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
            --outdir ${outdir}/pymol_images \
            --groups -1 \
            # --multisector_group_idxs 1 2 3
        ((count++))
        if [[ $count -eq $haltafter ]]; then
            break
        fi
    done
fi
